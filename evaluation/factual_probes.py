"""Corpus-grounded factual probe loading and scoring."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from inference.generate import GenerationConfig, generate
from inference.sampler import SamplerConfig
from model.gpt import GPT
from tokenizer.tokenizer import Tokenizer

DEFAULT_FAIR_SAMPLER = SamplerConfig(
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.3,
)


@dataclass(frozen=True)
class FactualProbe:
    """One factual completion probe with acceptable answer keywords."""

    prompt: str
    answers: tuple[str, ...]
    source: str = "unknown"


def load_probes(path: str | Path) -> list[FactualProbe]:
    """Load and validate probes from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        msg = f"Probe file {path} must contain a JSON array"
        raise TypeError(msg)

    probes: list[FactualProbe] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            msg = f"Probe entry {index} must be an object"
            raise TypeError(msg)
        prompt = entry.get("prompt")
        answers = entry.get("answers")
        if not isinstance(prompt, str) or not prompt.strip():
            msg = f"Probe entry {index} missing a non-empty 'prompt' string"
            raise ValueError(msg)
        if not isinstance(answers, list) or not answers:
            msg = f"Probe entry {index} missing a non-empty 'answers' list"
            raise ValueError(msg)
        answer_strings = tuple(str(answer) for answer in answers if str(answer).strip())
        if not answer_strings:
            msg = f"Probe entry {index} has no usable answer strings"
            raise ValueError(msg)
        source = str(entry.get("source", "unknown"))
        probes.append(FactualProbe(prompt=prompt, answers=answer_strings, source=source))
    return probes


def verify_in_corpus(
    probes: list[FactualProbe],
    documents: Iterable[str],
) -> tuple[list[FactualProbe], list[FactualProbe]]:
    """Return ``(verified, dropped)`` probes whose answers appear in ``documents``."""
    corpus = "\n".join(documents).lower()
    verified: list[FactualProbe] = []
    dropped: list[FactualProbe] = []
    for probe in probes:
        if any(answer.lower() in corpus for answer in probe.answers):
            verified.append(probe)
        else:
            dropped.append(probe)
    return verified, dropped


def completion_hits(completion: str, answers: Sequence[str]) -> bool:
    """Return True if any answer substring appears in ``completion``."""
    normalized = completion.lower()
    return any(answer.lower() in normalized for answer in answers)


def score_factual_probes(
    model: GPT,
    tokenizer: Tokenizer,
    probes: list[FactualProbe],
    *,
    sampler: SamplerConfig,
    device: torch.device,
    max_new_tokens: int = 48,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate one completion per probe and compute hit-rate."""
    was_training = model.training
    model.eval()
    per_probe: list[dict[str, Any]] = []
    hits = 0

    try:
        for probe in probes:
            torch.manual_seed(seed)
            result = generate(
                model,
                tokenizer,
                probe.prompt,
                GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    sampler=sampler,
                    add_bos=False,
                ),
                device=device,
            )
            completion = tokenizer.decode(result.token_ids, skip_special=True)
            hit = completion_hits(completion, probe.answers)
            hits += int(hit)
            per_probe.append(
                {
                    "prompt": probe.prompt,
                    "answers": list(probe.answers),
                    "source": probe.source,
                    "hit": hit,
                    "completion": completion,
                }
            )
    finally:
        if was_training:
            model.train()

    total = len(probes)
    hit_rate = (hits / total) if total else 0.0
    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "total": total,
        "sampler": {
            "temperature": sampler.temperature,
            "top_p": sampler.top_p,
            "top_k": sampler.top_k,
            "repetition_penalty": sampler.repetition_penalty,
        },
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "per_probe": per_probe,
    }


def write_probe_results(path: str | Path, results: dict[str, Any]) -> None:
    """Write probe scoring results to ``path`` as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
