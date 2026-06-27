"""Training loop with accumulation, AMP, clipping, eval, and logging."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from model.config import GPTConfig
from model.gpt import GPT, masked_cross_entropy
from training.checkpoint import load_checkpoint, save_checkpoint
from training.device import move_batch_to_device, resolve_device, resolve_dtype
from training.optimizer import AdamWConfig, create_adamw
from training.scheduler import CosineWarmupConfig, create_cosine_warmup_scheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Trainer hyperparameters and runtime options."""

    max_steps: int = 1000
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    seed: int = 42
    prefer_bf16: bool = True
    device: str | None = None
    stage_tag: str = "pretrain"
    tokenizer_path: str | None = None


@dataclass
class TrainMetrics:
    """Collected training metrics."""

    train_losses: list[float] = field(default_factory=list)
    eval_losses: list[tuple[int, float]] = field(default_factory=list)


class Trainer:
    """From-scratch GPT trainer with masked loss and reproducible checkpoints."""

    def __init__(
        self,
        model: GPT,
        config: GPTConfig,
        trainer_config: TrainerConfig,
        *,
        optimizer_config: AdamWConfig | None = None,
        scheduler_config: CosineWarmupConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.trainer_config = trainer_config
        self.device = resolve_device(trainer_config.device)
        self.dtype = resolve_dtype(self.device, prefer_bf16=trainer_config.prefer_bf16)
        self.use_amp = self.dtype == torch.bfloat16 and self.device.type != "cpu"

        self.model.to(self.device)
        if self.dtype == torch.bfloat16 and self.device.type != "cpu":
            self.model.to(dtype=self.dtype)

        optimizer_config = optimizer_config or AdamWConfig()
        scheduler_config = scheduler_config or CosineWarmupConfig(
            warmup_steps=min(10, trainer_config.max_steps // 10),
            max_steps=trainer_config.max_steps,
        )
        self.optimizer = create_adamw(self.model.parameters(), optimizer_config)
        self.scheduler = create_cosine_warmup_scheduler(self.optimizer, scheduler_config)
        self.global_step = 0
        self.metrics = TrainMetrics()

    def _autocast_context(self) -> contextlib.AbstractContextManager[Any]:
        """Return an autocast context for the active device, or a no-op on CPU."""
        if not self.use_amp:
            return contextlib.nullcontext()
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        if self.device.type == "mps":
            return torch.autocast(device_type="mps", dtype=self.dtype)
        return contextlib.nullcontext()

    def _forward_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute masked cross-entropy for one micro-batch."""
        batch = move_batch_to_device(batch, self.device)
        with self._autocast_context():
            logits = self.model(batch["input_ids"])
            return masked_cross_entropy(logits, batch["targets"], batch["loss_mask"])

    def train_micro_batch(self, batch: dict[str, torch.Tensor]) -> float:
        """Run forward/backward on one micro-batch (loss scaled for accumulation)."""
        scaled_loss = self._forward_loss(batch) / self.trainer_config.grad_accum_steps
        scaled_loss.backward()  # type: ignore[no-untyped-call]
        return float(scaled_loss.item() * self.trainer_config.grad_accum_steps)

    def clip_and_step(self) -> float:
        """Clip gradients, optimizer step, scheduler step; return grad norm."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.trainer_config.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.item())

    def optimizer_step_from_micro_batches(
        self,
        micro_batches: list[dict[str, torch.Tensor]],
    ) -> tuple[float, float]:
        """Accumulate gradients over micro-batches, then step."""
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch in micro_batches:
            total_loss += self.train_micro_batch(batch)
        grad_norm = self.clip_and_step()
        return total_loss / len(micro_batches), grad_norm

    def train_steps(
        self,
        micro_batches: list[dict[str, torch.Tensor]],
        *,
        num_steps: int,
        start_micro_index: int = 0,
        eval_batches: list[dict[str, torch.Tensor]] | None = None,
    ) -> list[float]:
        """Run ``num_steps`` optimizer steps from a fixed micro-batch stream."""
        accum = self.trainer_config.grad_accum_steps
        step_losses: list[float] = []
        micro_index = start_micro_index

        for _ in range(num_steps):
            if micro_index + accum > len(micro_batches):
                break
            chunks = micro_batches[micro_index : micro_index + accum]
            micro_index += accum
            loss, grad_norm = self.optimizer_step_from_micro_batches(chunks)
            self.global_step += 1
            step_losses.append(loss)
            self.metrics.train_losses.append(loss)

            if self.global_step % self.trainer_config.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "step=%d loss=%.4f grad_norm=%.4f lr=%.6f",
                    self.global_step,
                    loss,
                    grad_norm,
                    lr,
                )

            if (
                eval_batches is not None
                and self.global_step % self.trainer_config.eval_interval == 0
            ):
                eval_loss = self.evaluate(eval_batches)
                self.metrics.eval_losses.append((self.global_step, eval_loss))
                logger.info("step=%d eval_loss=%.4f", self.global_step, eval_loss)

        return step_losses

    @torch.no_grad()
    def evaluate(self, batches: list[dict[str, torch.Tensor]]) -> float:
        """Return mean eval loss over ``batches``."""
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        for batch in batches:
            total += float(self._forward_loss(batch).item())
        if was_training:
            self.model.train()
        return total / max(len(batches), 1)

    def save(self, path: str | Path) -> None:
        """Save a reproducible checkpoint."""
        save_checkpoint(
            path,
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            stage_tag=self.trainer_config.stage_tag,
            tokenizer_path=self.trainer_config.tokenizer_path,
        )

    def load(self, path: str | Path) -> int:
        """Load checkpoint and restore trainer state."""
        loaded = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.global_step = loaded.step
        return loaded.step


def clip_global_grad_norm(parameters: Any, max_norm: float) -> float:
    """Clip gradient global norm and return the pre-clip norm."""
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm).item())
