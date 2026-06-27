"""GPT model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Single source of truth for all model dimensions."""

    vocab_size: int = 32_000
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None
    d_model: int = 768
    ffn_mult: float = 8 / 3
    ctx_len: int = 2048
    rope_theta: float = 10_000.0
    rmsnorm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            msg = f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            raise ValueError(msg)
        kv_heads = self.num_kv_heads
        if self.n_head % kv_heads != 0:
            msg = f"n_head ({self.n_head}) must be divisible by n_kv_head ({kv_heads})"
            raise ValueError(msg)

    @property
    def head_dim(self) -> int:
        """Per-head dimension."""
        return self.d_model // self.n_head

    @property
    def num_kv_heads(self) -> int:
        """Number of key/value heads (equals ``n_head`` for MHA)."""
        return self.n_head if self.n_kv_head is None else self.n_kv_head

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head for grouped-query attention."""
        return self.n_head // self.num_kv_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """SwiGLU hidden dimension."""
        return int(self.ffn_mult * self.d_model)

    @classmethod
    def nano(cls, vocab_size: int = 512) -> GPTConfig:
        """Nano (~2M) config for wiring and overfit-one-batch tests."""
        return cls(
            vocab_size=vocab_size,
            n_layer=4,
            n_head=4,
            d_model=128,
            ctx_len=256,
        )

    @classmethod
    def tiny(cls, vocab_size: int = 4096) -> GPTConfig:
        """Tiny (~10M) config for Shakespeare pretraining (Phase 7)."""
        return cls(
            vocab_size=vocab_size,
            n_layer=6,
            n_head=6,
            d_model=384,
            ctx_len=512,
        )
