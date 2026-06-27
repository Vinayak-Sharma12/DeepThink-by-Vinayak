"""Pre-norm transformer block."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.config import GPTConfig
from model.kv_cache import KVCache
from model.rmsnorm import RMSNorm
from model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config)
        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Return block output with the same shape as ``x``."""
        attn_out = self.attention(
            self.attention_norm(x),
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
        )
        x = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(x))
        return cast(torch.Tensor, x + ffn_out)
