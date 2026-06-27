"""Token embedding layer."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from model.config import GPTConfig


class TokenEmbedding(nn.Module):
    """Token embedding lookup table."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids ``[B, T]`` to vectors ``[B, T, d_model]``."""
        return cast(torch.Tensor, self.embedding(input_ids))

    @property
    def weight(self) -> torch.Tensor:
        """Embedding matrix used for optional weight tying."""
        return self.embedding.weight
