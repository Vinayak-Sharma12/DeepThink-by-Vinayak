"""SwiGLU feed-forward network."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig


class SwiGLU(nn.Module):
    """SwiGLU MLP block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = config.ffn_hidden_dim
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return SwiGLU output with the same shape as ``x``."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return cast(torch.Tensor, self.down_proj(gate * up))
