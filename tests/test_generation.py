"""Unit tests for text generation and sampling."""

from __future__ import annotations

import importlib

import pytest
import torch

import inference.generate as generate_module
from inference.generate import GenerationConfig, apply_thinking_mode
from inference.sampler import SamplerConfig, apply_repetition_penalty, apply_top_k, sample_token
from model import GPT, GPTConfig
from tokenizer import Tokenizer

generate = generate_module.generate


def _tiny_model_and_tokenizer() -> tuple[GPT, Tokenizer]:
    torch.manual_seed(0)
    tokenizer = Tokenizer.train_sample(vocab_size=512)
    config = GPTConfig(
        vocab_size=512,
        n_layer=2,
        n_head=2,
        d_model=32,
        ctx_len=64,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    return model, tokenizer


def test_greedy_kv_cache_matches_no_cache() -> None:
    """Greedy decoding with KV cache matches token-for-token without cache."""
    model, tokenizer = _tiny_model_and_tokenizer()
    prompt = "Virtue is"
    greedy = SamplerConfig(temperature=0.0)

    torch.manual_seed(1)
    cached = generate(
        model,
        tokenizer,
        prompt,
        GenerationConfig(max_new_tokens=16, sampler=greedy),
        use_kv_cache=True,
    )

    torch.manual_seed(1)
    no_cache = generate(
        model,
        tokenizer,
        prompt,
        GenerationConfig(max_new_tokens=16, sampler=greedy),
        use_kv_cache=False,
    )

    assert cached.token_ids == no_cache.token_ids


def test_generation_stops_on_eos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generation stops when the sampler returns the EOS token."""
    model, tokenizer = _tiny_model_and_tokenizer()
    eos_id = tokenizer.vocab.special_token_id("<|eos|>")

    def _always_eos(_logits: torch.Tensor, _config: SamplerConfig, **_: object) -> int:
        return eos_id

    monkeypatch.setattr(generate_module, "sample_token", _always_eos)
    result = generate(
        model,
        tokenizer,
        "Hello",
        GenerationConfig(max_new_tokens=50),
    )
    assert result.token_ids == []


def test_max_new_tokens_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generation never exceeds ``max_new_tokens`` when EOS is not emitted."""
    model, tokenizer = _tiny_model_and_tokenizer()
    eos_id = tokenizer.vocab.special_token_id("<|eos|>")

    def _never_eos(_logits: torch.Tensor, _config: SamplerConfig, **_: object) -> int:
        return 3 if eos_id != 3 else 4

    monkeypatch.setattr(generate_module, "sample_token", _never_eos)
    max_new = 7
    result = generate(
        model,
        tokenizer,
        "Cap test",
        GenerationConfig(max_new_tokens=max_new),
    )
    assert len(result.token_ids) == max_new


def test_thinking_hide_removes_span() -> None:
    """Hide mode strips the think span but preserves it in ``thinking``."""
    text = "Answer.<|think|>hidden trace<|/think|> Done."
    visible, thinking = apply_thinking_mode(text, "hide")
    assert "<|think|>" not in visible
    assert thinking == "hidden trace"
    assert visible == "Answer. Done."


def test_thinking_show_preserves_span() -> None:
    """Show mode keeps the full decoded text."""
    text = "Answer.<|think|>trace<|/think|>"
    visible, thinking = apply_thinking_mode(text, "show")
    assert visible == text
    assert thinking == "trace"


def test_top_k_limits_support() -> None:
    """Top-k zeroes logits outside the top-k support."""
    logits = torch.tensor([1.0, 2.0, 3.0, 0.5])
    filtered = apply_top_k(logits, top_k=2)
    assert filtered[2].item() != float("-inf")
    assert filtered[0].item() == float("-inf")


def test_temperature_sampling_returns_valid_token() -> None:
    """Stochastic sampling returns an in-vocab token id."""
    torch.manual_seed(5)
    logits = torch.randn(128)
    token_id = sample_token(logits, SamplerConfig(temperature=1.0, top_k=10))
    assert 0 <= token_id < 128


def test_greedy_sampling_is_argmax() -> None:
    """Greedy sampling picks the argmax logit."""
    logits = torch.tensor([0.1, 2.0, 1.0])
    token_id = sample_token(logits, SamplerConfig(temperature=0.0))
    assert token_id == 1


def test_repetition_penalty_is_noop_at_one() -> None:
    """Repetition penalty 1.0 leaves logits unchanged."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    adjusted = apply_repetition_penalty(logits, [0, 2], 1.0)
    assert torch.equal(adjusted, logits)


def test_repetition_penalty_downweights_seen_token() -> None:
    """Seen tokens are down-weighted when penalty exceeds 1."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    adjusted = apply_repetition_penalty(logits, [2], 2.0)
    assert adjusted[2].item() == pytest.approx(1.5)
    assert adjusted[0].item() == pytest.approx(1.0)
    assert adjusted[1].item() == pytest.approx(2.0)

    token_id = sample_token(
        logits,
        SamplerConfig(temperature=0.0, repetition_penalty=2.0),
        prev_token_ids=[2],
    )
    assert token_id == 1


def test_generate_module_is_submodule() -> None:
    """The implementation lives in the ``inference.generate`` submodule."""
    module = importlib.import_module("inference.generate")
    assert hasattr(module, "generate")
    assert callable(module.generate)
