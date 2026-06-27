"""Root-mean-square layer normalization."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import GPTConfig


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model))
        self.eps = config.rmsnorm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension by root-mean-square."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight
