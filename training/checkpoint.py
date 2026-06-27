"""Checkpoint save/load with full reproducibility state."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from model.config import GPTConfig


@dataclass
class RNGState:
    """Captured RNG state for python, numpy, and torch."""

    python: tuple[Any, ...]
    numpy: Any
    torch_rng: torch.Tensor
    torch_cuda: list[torch.Tensor] | None = None


def capture_rng_state() -> RNGState:
    """Capture RNG state from python, numpy, and torch."""
    torch_cuda: list[torch.Tensor] | None = None
    if torch.cuda.is_available():
        torch_cuda = torch.cuda.get_rng_state_all()
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        torch_cuda=torch_cuda,
    )


def restore_rng_state(state: RNGState) -> None:
    """Restore RNG state captured by ``capture_rng_state``."""
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch_rng = state.torch_rng
    if not isinstance(torch_rng, torch.Tensor):
        torch_rng = torch.as_tensor(torch_rng, dtype=torch.uint8)
    else:
        torch_rng = torch_rng.detach().cpu().to(torch.uint8)
    torch.set_rng_state(torch_rng)
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state_to_dict(state: RNGState) -> dict[str, Any]:
    """Serialize ``RNGState`` for checkpoint storage."""
    return {
        "python": state.python,
        "numpy": state.numpy,
        "torch_rng": state.torch_rng,
        "torch_cuda": state.torch_cuda,
    }


def rng_state_from_dict(data: dict[str, Any]) -> RNGState:
    """Deserialize ``RNGState`` from a checkpoint dict."""
    torch_rng = data.get("torch_rng", data.get("torch"))
    if torch_rng is None:
        msg = "Checkpoint rng_state missing torch_rng"
        raise KeyError(msg)
    return RNGState(
        python=data["python"],
        numpy=data["numpy"],
        torch_rng=torch_rng,
        torch_cuda=data.get("torch_cuda"),
    )


def gpt_config_to_dict(config: GPTConfig) -> dict[str, Any]:
    """Serialize ``GPTConfig`` to a plain dict."""
    return asdict(config)


def gpt_config_from_dict(data: dict[str, Any]) -> GPTConfig:
    """Deserialize ``GPTConfig`` from a checkpoint dict."""
    return GPTConfig(**data)


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: GPTConfig,
    stage_tag: str,
    tokenizer_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist model, optimizer, scheduler, step, RNG, and metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": rng_state_to_dict(capture_rng_state()),
        "gpt_config": gpt_config_to_dict(config),
        "stage_tag": stage_tag,
        "tokenizer_path": tokenizer_path,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, output_path)


@dataclass
class LoadedCheckpoint:
    """Checkpoint contents returned by ``load_checkpoint``."""

    step: int
    config: GPTConfig
    stage_tag: str
    tokenizer_path: str | None
    extra: dict[str, Any] | None


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Restore model, optimizer, scheduler, and RNG state from disk."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(rng_state_from_dict(payload["rng_state"]))
    return LoadedCheckpoint(
        step=int(payload["step"]),
        config=gpt_config_from_dict(payload["gpt_config"]),
        stage_tag=str(payload["stage_tag"]),
        tokenizer_path=payload.get("tokenizer_path"),
        extra=payload.get("extra"),
    )
