"""Learning-rate schedulers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class CosineWarmupConfig:
    """Cosine decay with linear warmup."""

    warmup_steps: int
    max_steps: int
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            msg = f"warmup_steps must be non-negative, got {self.warmup_steps}"
            raise ValueError(msg)
        if self.max_steps <= 0:
            msg = f"max_steps must be positive, got {self.max_steps}"
            raise ValueError(msg)
        if self.warmup_steps > self.max_steps:
            msg = "warmup_steps cannot exceed max_steps"
            raise ValueError(msg)


def lr_multiplier(step: int, config: CosineWarmupConfig) -> float:
    """Return the LR multiplier at optimizer step ``step`` (0-indexed)."""
    if step < config.warmup_steps:
        if config.warmup_steps == 0:
            return 1.0
        return float(step + 1) / float(config.warmup_steps)
    if config.max_steps == config.warmup_steps:
        return config.min_lr_ratio
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def create_cosine_warmup_scheduler(
    optimizer: Optimizer,
    config: CosineWarmupConfig,
) -> LambdaLR:
    """Create a ``LambdaLR`` scheduler with linear warmup and cosine decay."""
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(step, config),
    )
