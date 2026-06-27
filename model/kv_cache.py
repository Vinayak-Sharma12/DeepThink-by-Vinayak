"""Key/value cache for incremental decoding."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    """Per-layer key/value tensors for autoregressive decoding."""

    keys: list[torch.Tensor | None] = field(default_factory=list)
    values: list[torch.Tensor | None] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layer: int) -> KVCache:
        """Create an empty cache for ``n_layer`` transformer blocks."""
        return cls(keys=[None] * n_layer, values=[None] * n_layer)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full cached sequence."""
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            prior_key = self.keys[layer_idx]
            prior_value = self.values[layer_idx]
            assert prior_key is not None
            assert prior_value is not None
            self.keys[layer_idx] = torch.cat([prior_key, key], dim=2)
            self.values[layer_idx] = torch.cat([prior_value, value], dim=2)
        cached_key = self.keys[layer_idx]
        cached_value = self.values[layer_idx]
        assert cached_key is not None
        assert cached_value is not None
        return cached_key, cached_value
