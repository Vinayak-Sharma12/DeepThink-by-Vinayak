"""AdamW optimizer configuration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AdamWConfig:
    """Hyperparameters for decoupled-weight-decay AdamW."""

    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


def create_adamw(
    parameters: Iterable[nn.Parameter],
    config: AdamWConfig,
) -> torch.optim.AdamW:
    """Build a PyTorch AdamW optimizer from ``config``."""
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
