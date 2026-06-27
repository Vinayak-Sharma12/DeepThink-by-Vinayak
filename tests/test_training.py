"""Unit tests for the from-scratch training framework."""

from __future__ import annotations

from pathlib import Path

import torch

from model import GPT, GPTConfig
from training.checkpoint import load_checkpoint, seed_everything
from training.scheduler import CosineWarmupConfig, lr_multiplier
from training.trainer import Trainer, TrainerConfig, clip_global_grad_norm


def _nano_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=2,
        d_model=32,
        ctx_len=32,
        dropout=0.0,
    )


def _make_micro_batches(
    count: int,
    *,
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 64,
    seed: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Build deterministic micro-batches for training tests."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    batches: list[dict[str, torch.Tensor]] = []
    for _ in range(count):
        input_ids = torch.randint(
            1,
            vocab_size,
            (batch_size, seq_len),
            generator=generator,
        )
        targets = torch.randint(
            1,
            vocab_size,
            (batch_size, seq_len),
            generator=generator,
        )
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        batches.append(
            {
                "input_ids": input_ids,
                "targets": targets,
                "loss_mask": loss_mask,
            }
        )
    return batches


def _trainer(model: GPT, config: GPTConfig, **kwargs: object) -> Trainer:
    trainer_config = TrainerConfig(
        device="cpu",
        prefer_bf16=False,
        log_interval=1000,
        eval_interval=1000,
        seed=42,
        **kwargs,  # type: ignore[arg-type]
    )
    return Trainer(
        model,
        config,
        trainer_config,
        scheduler_config=CosineWarmupConfig(warmup_steps=2, max_steps=100),
    )


def test_loss_decreases_on_synthetic_task() -> None:
    """Training loss trends downward on a tiny repeatable task."""
    seed_everything(42)
    config = _nano_config()
    model = GPT(config)
    trainer = _trainer(model, config, max_steps=50, grad_accum_steps=1)
    micro_batches = _make_micro_batches(60, batch_size=2, vocab_size=config.vocab_size)
    losses = trainer.train_steps(micro_batches, num_steps=40)
    assert losses[-1] < losses[0]


def test_grad_accumulation_equivalence() -> None:
    """K accumulated micro-steps matches one K-sized batch update."""
    config = _nano_config()
    micro_one = _make_micro_batches(4, batch_size=1, seed=7)
    micro_four = [
        {
            "input_ids": torch.cat([b["input_ids"] for b in micro_one], dim=0),
            "targets": torch.cat([b["targets"] for b in micro_one], dim=0),
            "loss_mask": torch.cat([b["loss_mask"] for b in micro_one], dim=0),
        }
    ]

    seed_everything(123)
    model_accum = GPT(config)
    trainer_accum = _trainer(model_accum, config, grad_accum_steps=4)
    trainer_accum.optimizer_step_from_micro_batches(micro_one)

    seed_everything(123)
    model_single = GPT(config)
    trainer_single = _trainer(model_single, config, grad_accum_steps=1)
    trainer_single.optimizer_step_from_micro_batches(micro_four)

    for param_accum, param_single in zip(
        model_accum.parameters(),
        model_single.parameters(),
        strict=True,
    ):
        assert torch.allclose(param_accum, param_single, atol=1e-5, rtol=1e-4)


def test_grad_clipping_caps_global_norm() -> None:
    """Global grad-norm clipping enforces the configured maximum."""
    config = _nano_config()
    model = GPT(config)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    grad_norm = clip_global_grad_norm(model.parameters(), max_norm=1.0)
    assert grad_norm > 1.0

    total_sq = 0.0
    for parameter in model.parameters():
        assert parameter.grad is not None
        total_sq += float(parameter.grad.pow(2).sum().item())
    clipped_norm = total_sq**0.5
    assert clipped_norm <= 1.0 + 1e-6


def test_cosine_warmup_scheduler() -> None:
    """Warmup increases LR multiplier; post-warmup cosine decays."""
    config = CosineWarmupConfig(warmup_steps=4, max_steps=20, min_lr_ratio=0.1)
    assert lr_multiplier(0, config) < lr_multiplier(3, config)
    assert lr_multiplier(19, config) <= lr_multiplier(4, config)
    assert lr_multiplier(19, config) >= config.min_lr_ratio


def test_bit_reproducible_resume(tmp_path: Path) -> None:
    """Checkpoint resume reproduces the uninterrupted loss curve."""
    config = _nano_config()
    micro_batches = _make_micro_batches(80, batch_size=2, vocab_size=config.vocab_size, seed=99)
    checkpoint_path = tmp_path / "checkpoint.pt"

    seed_everything(7)
    model_full = GPT(config)
    trainer_full = _trainer(model_full, config, grad_accum_steps=1)
    losses_full = trainer_full.train_steps(micro_batches, num_steps=20)

    seed_everything(7)
    model_resume = GPT(config)
    trainer_resume = _trainer(model_resume, config, grad_accum_steps=1)
    losses_first = trainer_resume.train_steps(micro_batches, num_steps=10)
    trainer_resume.save(checkpoint_path)

    seed_everything(7)
    model_reload = GPT(config)
    trainer_reload = _trainer(model_reload, config, grad_accum_steps=1)
    trainer_reload.train_steps(micro_batches, num_steps=10)
    loaded_step = trainer_reload.load(checkpoint_path)
    assert loaded_step == 10
    losses_second = trainer_reload.train_steps(
        micro_batches,
        num_steps=10,
        start_micro_index=10,
    )

    assert losses_first == losses_full[:10]
    assert losses_second == losses_full[10:20]


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """Saved checkpoint restores model and optimizer state."""
    config = _nano_config()
    seed_everything(0)
    model = GPT(config)
    trainer = _trainer(model, config)
    micro_batches = _make_micro_batches(5, batch_size=1, vocab_size=config.vocab_size)
    trainer.train_steps(micro_batches, num_steps=3)
    path = tmp_path / "ckpt.pt"
    trainer.save(path)

    seed_everything(999)
    model_new = GPT(config)
    trainer_new = _trainer(model_new, config)
    loaded = load_checkpoint(
        path,
        model=model_new,
        optimizer=trainer_new.optimizer,
        scheduler=trainer_new.scheduler,
    )
    assert loaded.step == 3
    assert loaded.stage_tag == "pretrain"
    for original, restored in zip(model.parameters(), model_new.parameters(), strict=True):
        assert torch.equal(original, restored)
