"""KV-cache helpers for incremental text generation."""

from __future__ import annotations

from model.kv_cache import KVCache

__all__ = ["KVCache"]


def empty_kv_cache(n_layer: int) -> KVCache:
    """Create an empty cache for a model with ``n_layer`` blocks."""
    return KVCache.empty(n_layer)
