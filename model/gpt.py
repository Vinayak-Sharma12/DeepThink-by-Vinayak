"""Full decoder-only GPT model."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.block import TransformerBlock
from model.config import GPTConfig
from model.embeddings import TokenEmbedding
from model.kv_cache import KVCache
from model.rmsnorm import RMSNorm


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy averaged over unmasked positions."""
    batch, seq_len, vocab = logits.shape
    per_token = F.cross_entropy(
        logits.reshape(batch * seq_len, vocab),
        targets.reshape(batch * seq_len),
        reduction="none",
    ).view(batch, seq_len)
    mask = loss_mask.to(dtype=per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters, avoiding double-count for tied weights."""
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        param_id = id(parameter)
        if param_id in seen:
            continue
        seen.add(param_id)
        total += parameter.numel()
    return total


def estimate_parameter_count(config: GPTConfig) -> int:
    """Hand-count parameters for a config (used in tests)."""
    d_model = config.d_model
    vocab = config.vocab_size
    hidden = config.ffn_hidden_dim
    n_kv = config.num_kv_heads
    head_dim = config.head_dim

    embed = vocab * d_model
    lm_head = 0 if config.tie_weights else vocab * d_model

    q = d_model * config.n_head * head_dim
    k = d_model * n_kv * head_dim
    v = d_model * n_kv * head_dim
    o = config.n_head * head_dim * d_model
    attn = q + k + v + o

    ffn = 3 * d_model * hidden
    block_norms = 2 * d_model
    per_layer = attn + ffn + block_norms

    return embed + lm_head + config.n_layer * per_layer + d_model


class GPT(nn.Module):
    """Decoder-only GPT with RoPE, RMSNorm, SwiGLU, GQA, and KV cache."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.final_norm = RMSNorm(config)
        self.lm_head: nn.Linear | None
        if not config.tie_weights:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Return logits with shape ``[batch, seq, vocab_size]``."""
        hidden = self.token_embedding(input_ids)
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                start_pos=start_pos,
            )
        hidden = self.final_norm(hidden)
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = F.linear(hidden, self.token_embedding.weight)
        return cast(torch.Tensor, logits)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return count_parameters(self)

    @torch.no_grad()
    def forward_logits_with_cache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run incremental decoding with KV cache and return full-sequence logits."""
        self.eval()
        batch, seq_len = input_ids.shape
        cache = KVCache.empty(self.config.n_layer)
        logits_steps: list[torch.Tensor] = []
        for position in range(seq_len):
            step_logits = self(
                input_ids[:, position : position + 1],
                kv_cache=cache,
                start_pos=position,
            )
            logits_steps.append(step_logits)
        return torch.cat(logits_steps, dim=1)
