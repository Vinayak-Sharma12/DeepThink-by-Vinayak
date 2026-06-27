"""High-level text generation with KV-cache decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import torch

from inference.kvcache import empty_kv_cache
from inference.sampler import SamplerConfig, sample_token
from model.gpt import GPT
from tokenizer.tokenizer import Tokenizer

THINK_OPEN = "<|think|>"
THINK_CLOSE = "<|/think|>"
ThinkMode = Literal["show", "hide"]


@dataclass
class GenerationConfig:
    """Generation runtime options."""

    max_new_tokens: int = 256
    sampler: SamplerConfig | None = None
    thinking_mode: ThinkMode = "show"
    add_bos: bool = True
    stop_on_eos: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            msg = f"max_new_tokens must be positive, got {self.max_new_tokens}"
            raise ValueError(msg)
        if self.sampler is None:
            self.sampler = SamplerConfig()


@dataclass
class GenerationResult:
    """Generated text and metadata."""

    token_ids: list[int]
    prompt_token_ids: list[int]
    raw_text: str
    text: str
    thinking: str | None = None


def extract_thinking_span(text: str) -> tuple[str, str | None]:
    """Return ``(text_without_think_span, think_content_or_none)``."""
    pattern = re.compile(
        re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        return text, None
    thinking = match.group(1)
    visible = (text[: match.start()] + text[match.end() :]).strip()
    return visible, thinking


def apply_thinking_mode(text: str, mode: ThinkMode) -> tuple[str, str | None]:
    """Apply show/hide policy for the reasoning span."""
    visible, thinking = extract_thinking_span(text)
    if mode == "hide":
        return visible, thinking
    return text, thinking


def _decode_completion(
    tokenizer: Tokenizer,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
) -> str:
    """Decode prompt + completion, keeping special tokens visible."""
    all_ids = prompt_token_ids + completion_token_ids
    return tokenizer.decode(all_ids, skip_special=False)


def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
    *,
    use_kv_cache: bool = True,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    """Generate completion tokens for ``prompt``."""
    config = config or GenerationConfig()
    sampler = config.sampler or SamplerConfig()
    device = device or torch.device("cpu")
    eos_id = tokenizer.vocab.special_token_id("<|eos|>")

    prompt_ids = tokenizer.encode(prompt, add_bos=config.add_bos, add_eos=False)
    generated: list[int] = []

    was_training = model.training
    model.eval()
    try:
        if use_kv_cache:
            generated = _generate_with_cache(
                model,
                prompt_ids,
                config.max_new_tokens,
                sampler,
                eos_id=eos_id,
                stop_on_eos=config.stop_on_eos,
                device=device,
                generator=generator,
            )
        else:
            generated = _generate_without_cache(
                model,
                prompt_ids,
                config.max_new_tokens,
                sampler,
                eos_id=eos_id,
                stop_on_eos=config.stop_on_eos,
                device=device,
                generator=generator,
            )
    finally:
        if was_training:
            model.train()

    raw_text = _decode_completion(tokenizer, prompt_ids, generated)
    text, thinking = apply_thinking_mode(raw_text, config.thinking_mode)
    return GenerationResult(
        token_ids=generated,
        prompt_token_ids=prompt_ids,
        raw_text=raw_text,
        text=text,
        thinking=thinking,
    )


def _generate_with_cache(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    sampler: SamplerConfig,
    *,
    eos_id: int,
    stop_on_eos: bool,
    device: torch.device,
    generator: torch.Generator | None,
) -> list[int]:
    """Incremental decoding with KV cache."""
    cache = empty_kv_cache(model.config.n_layer)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = model(prompt_tensor, kv_cache=cache, start_pos=0)
    next_logits = logits[0, -1, :]

    generated: list[int] = []
    start_pos = len(prompt_ids)

    for offset in range(max_new_tokens):
        token_id = sample_token(next_logits, sampler, generator=generator)
        if stop_on_eos and token_id == eos_id:
            break
        generated.append(token_id)
        step_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        step_logits = model(
            step_tensor,
            kv_cache=cache,
            start_pos=start_pos + offset,
        )
        next_logits = step_logits[0, -1, :]

    return generated


def _generate_without_cache(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    sampler: SamplerConfig,
    *,
    eos_id: int,
    stop_on_eos: bool,
    device: torch.device,
    generator: torch.Generator | None,
) -> list[int]:
    """Autoregressive decoding without KV cache (used for parity checks)."""
    sequence = list(prompt_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
        logits = model(input_tensor)
        next_logits = logits[0, -1, :]
        token_id = sample_token(next_logits, sampler, generator=generator)
        if stop_on_eos and token_id == eos_id:
            break
        generated.append(token_id)
        sequence.append(token_id)

    return generated
