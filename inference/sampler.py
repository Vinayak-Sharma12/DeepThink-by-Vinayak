"""Logits processing and token sampling strategies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplerConfig:
    """Sampling hyperparameters for next-token selection."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            msg = f"temperature must be non-negative, got {self.temperature}"
            raise ValueError(msg)
        if self.top_k is not None and self.top_k <= 0:
            msg = f"top_k must be positive, got {self.top_k}"
            raise ValueError(msg)
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            msg = f"top_p must be in (0, 1], got {self.top_p}"
            raise ValueError(msg)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature (greedy when ``temperature == 0``)."""
    if temperature == 0.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask logits outside the top-k set."""
    vocab = logits.shape[-1]
    k = min(top_k, vocab)
    top_values, _ = torch.topk(logits, k)
    threshold = top_values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative - sorted_probs > top_p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, sorted_indices, sorted_logits)


def make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    """Return a device-matched generator for reproducible sampling."""
    if seed <= 0:
        return None
    torch.manual_seed(seed)
    if device.type in ("cpu", "cuda", "mps"):
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed)
        return generator
    return None


def sample_token(
    logits: torch.Tensor,
    config: SamplerConfig,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from ``logits`` ``[vocab]`` using ``config``."""
    if logits.ndim != 1:
        msg = f"Expected 1-D logits, got shape {tuple(logits.shape)}"
        raise ValueError(msg)

    if config.temperature == 0.0 and config.top_k is None and config.top_p is None:
        return int(torch.argmax(logits).item())

    processed = apply_temperature(logits, config.temperature)
    if config.top_k is not None:
        processed = apply_top_k(processed, config.top_k)
    if config.top_p is not None:
        processed = apply_top_p(processed, config.top_p)

    if config.temperature == 0.0:
        return int(torch.argmax(processed).item())

    probabilities = F.softmax(processed, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
