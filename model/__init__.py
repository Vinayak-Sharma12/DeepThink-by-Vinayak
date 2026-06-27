"""Project LOGOS — GPT architecture (RoPE, RMSNorm, SwiGLU, attention, blocks)."""

from model.attention import CausalSelfAttention
from model.block import TransformerBlock
from model.config import GPTConfig
from model.embeddings import TokenEmbedding
from model.gpt import GPT, count_parameters, estimate_parameter_count, masked_cross_entropy
from model.kv_cache import KVCache
from model.rmsnorm import RMSNorm
from model.rope import RotaryEmbedding, apply_rotary_emb, rotate_half
from model.swiglu import SwiGLU

__all__ = [
    "CausalSelfAttention",
    "GPT",
    "GPTConfig",
    "KVCache",
    "RotaryEmbedding",
    "RMSNorm",
    "SwiGLU",
    "TokenEmbedding",
    "TransformerBlock",
    "apply_rotary_emb",
    "count_parameters",
    "estimate_parameter_count",
    "masked_cross_entropy",
    "rotate_half",
]
