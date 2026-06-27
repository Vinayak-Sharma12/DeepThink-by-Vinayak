"""Unit tests for the GPT architecture modules."""

from __future__ import annotations

import torch

from model import (
    GPT,
    GPTConfig,
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TokenEmbedding,
    TransformerBlock,
    count_parameters,
    estimate_parameter_count,
    masked_cross_entropy,
    rotate_half,
)
from model.attention import CausalSelfAttention


def _tiny_config(*, n_kv_head: int | None = None) -> GPTConfig:
    return GPTConfig(
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=n_kv_head,
        d_model=64,
        ctx_len=128,
        dropout=0.0,
    )


def test_rmsnorm_shape_and_backward() -> None:
    """RMSNorm preserves shape and supports backward."""
    config = _tiny_config()
    norm = RMSNorm(config)
    x = torch.randn(2, 8, config.d_model, requires_grad=True)
    out = norm(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_swiglu_shape_and_backward() -> None:
    """SwiGLU preserves shape and supports backward."""
    config = _tiny_config()
    ffn = SwiGLU(config)
    x = torch.randn(2, 8, config.d_model, requires_grad=True)
    out = ffn(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_token_embedding_shape() -> None:
    """Token embedding maps ids to vectors."""
    config = _tiny_config()
    embed = TokenEmbedding(config)
    ids = torch.randint(0, config.vocab_size, (2, 10))
    vectors = embed(ids)
    assert vectors.shape == (2, 10, config.d_model)


def test_attention_shape_without_cache() -> None:
    """Attention output matches input shape without KV cache."""
    config = _tiny_config()
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 8, config.d_model)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_shape_with_cache() -> None:
    """Attention supports incremental decoding with KV cache."""
    config = _tiny_config()
    attn = CausalSelfAttention(config)
    cache = KVCache.empty(n_layer=1)
    x = torch.randn(2, 1, config.d_model)
    out = attn(x, kv_cache=cache, layer_idx=0, start_pos=0)
    assert out.shape == x.shape
    x2 = torch.randn(2, 1, config.d_model)
    out2 = attn(x2, kv_cache=cache, layer_idx=0, start_pos=1)
    assert out2.shape == x2.shape


def test_gqa_attention_shape() -> None:
    """Grouped-query attention runs with fewer KV heads."""
    config = _tiny_config(n_kv_head=2)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 8, config.d_model)
    out = attn(x)
    assert out.shape == x.shape


def test_transformer_block_shape() -> None:
    """Transformer block preserves tensor shape."""
    config = _tiny_config()
    block = TransformerBlock(config)
    x = torch.randn(2, 8, config.d_model)
    out = block(x)
    assert out.shape == x.shape


def test_gpt_forward_logits_shape() -> None:
    """GPT maps input ids to logits ``[B, T, vocab]``."""
    config = _tiny_config()
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    logits = model(input_ids)
    assert logits.shape == (2, 12, config.vocab_size)


def test_rope_relative_position_invariance() -> None:
    """RoPE dot product depends on relative, not absolute, position."""
    config = _tiny_config()
    rope = RotaryEmbedding(config)
    head_dim = config.head_dim
    query = torch.randn(1, 1, 1, head_dim)
    key = torch.randn(1, 1, 1, head_dim)

    def rotated_dot(pos_q: int, pos_k: int) -> torch.Tensor:
        cos_q = rope.cos[:, :, pos_q : pos_q + 1, :].to(query.dtype)
        sin_q = rope.sin[:, :, pos_q : pos_q + 1, :].to(query.dtype)
        cos_k = rope.cos[:, :, pos_k : pos_k + 1, :].to(key.dtype)
        sin_k = rope.sin[:, :, pos_k : pos_k + 1, :].to(key.dtype)
        rot_q = (query * cos_q) + (rotate_half(query) * sin_q)
        rot_k = (key * cos_k) + (rotate_half(key) * sin_k)
        return (rot_q * rot_k).sum()

    offset = 3
    first = rotated_dot(2, 2 + offset)
    second = rotated_dot(17, 17 + offset)
    assert torch.allclose(first, second, atol=1e-5)


def test_kv_cache_logits_parity() -> None:
    """Incremental KV-cache decoding matches full forward logits."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = GPT(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    logits_no_cache = model(input_ids)
    logits_cached = model.forward_logits_with_cache(input_ids)
    assert torch.allclose(logits_no_cache, logits_cached, atol=1e-5, rtol=1e-4)


def test_count_parameters_matches_estimate() -> None:
    """Parameter counter matches a hand-derived total."""
    config = GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        ctx_len=64,
        tie_weights=True,
    )
    model = GPT(config)
    assert count_parameters(model) == estimate_parameter_count(config)


def test_count_parameters_without_weight_tying() -> None:
    """Untied models count separate lm_head parameters."""
    config = GPTConfig(
        vocab_size=100,
        n_layer=1,
        n_head=2,
        d_model=32,
        ctx_len=64,
        tie_weights=False,
    )
    model = GPT(config)
    assert count_parameters(model) == estimate_parameter_count(config)


def test_overfit_one_batch() -> None:
    """Nano model memorizes a single batch (loss -> ~0)."""
    torch.manual_seed(42)
    config = GPTConfig.nano(vocab_size=128)
    config.dropout = 0.0
    model = GPT(config)
    model.train()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    final_loss = float("inf")
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = masked_cross_entropy(logits, targets, loss_mask)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < 0.05, f"Expected near-zero loss, got {final_loss:.4f}"


def test_masked_cross_entropy_respects_mask() -> None:
    """Loss utility ignores masked-out positions."""
    logits = torch.zeros(1, 3, 4)
    logits[0, 1, 2] = 10.0
    targets = torch.tensor([[0, 2, 1]])
    full_mask = torch.ones(1, 3)
    zero_mask = torch.tensor([[0.0, 0.0, 0.0]])

    full_loss = masked_cross_entropy(logits, targets, full_mask)
    zero_loss = masked_cross_entropy(logits, targets, zero_mask)
    assert full_loss.item() > 0.0
    assert zero_loss.item() == 0.0
