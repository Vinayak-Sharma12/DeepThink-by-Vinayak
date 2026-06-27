"""Rotary positional embeddings (RoPE)."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import GPTConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    ``query`` / ``key`` shape: ``[batch, n_head, seq, head_dim]``.
    ``cos`` / ``sin`` shape: ``[1, 1, seq, head_dim]``.
    """
    rotated_q = (query * cos) + (rotate_half(query) * sin)
    rotated_k = (key * cos) + (rotate_half(key) * sin)
    return rotated_q, rotated_k


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables up to ``ctx_len``."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        head_dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(config.ctx_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE starting at ``start_pos`` for incremental decoding."""
        seq_len = query.shape[2]
        cos = self.cos[:, :, start_pos : start_pos + seq_len, :].to(query.dtype)
        sin = self.sin[:, :, start_pos : start_pos + seq_len, :].to(query.dtype)
        return apply_rotary_emb(query, key, cos, sin)

    def rotate_queries(self, query: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply RoPE to queries only (used in relative-position tests)."""
        seq_len = query.shape[2]
        cos = self.cos[:, :, start_pos : start_pos + seq_len, :].to(query.dtype)
        sin = self.sin[:, :, start_pos : start_pos + seq_len, :].to(query.dtype)
        return (query * cos) + (rotate_half(query) * sin)
