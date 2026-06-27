"""Causal multi-head self-attention with optional GQA and KV cache."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig
from model.kv_cache import KVCache
from model.rope import RotaryEmbedding


def repeat_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    n_rep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat KV heads to match the number of query heads for GQA."""
    if n_rep == 1:
        return key, value
    batch, n_kv_head, seq_len, head_dim = key.shape
    key = key[:, :, None, :, :].expand(batch, n_kv_head, n_rep, seq_len, head_dim)
    value = value[:, :, None, :, :].expand(batch, n_kv_head, n_rep, seq_len, head_dim)
    key = key.reshape(batch, n_kv_head * n_rep, seq_len, head_dim)
    value = value.reshape(batch, n_kv_head * n_rep, seq_len, head_dim)
    return key, value


class CausalSelfAttention(nn.Module):
    """Pre-norm causal self-attention with RoPE, GQA, and KV cache support."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_kv_groups

        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        self.q_proj = nn.Linear(config.d_model, q_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Return attention output with shape ``[B, T, d_model]``."""
        batch, seq_len, _model_dim = x.shape

        query = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        query, key = self.rope(query, key, start_pos=start_pos)

        total_seq_len = key.shape[2]
        if kv_cache is not None:
            key, value = kv_cache.update(layer_idx, key, value)
            total_seq_len = key.shape[2]

        key, value = repeat_kv(key, value, self.n_rep)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if kv_cache is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, total_seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        else:
            # Incremental decoding: queries attend to all cached keys (already causal).
            pass

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.n_head * self.head_dim)
        )
        return cast(torch.Tensor, self.o_proj(context))
