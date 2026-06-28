"""Unit tests for corpus-grounded factual probe scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.factual_probes import (
    FactualProbe,
    completion_hits,
    load_probes,
    score_factual_probes,
    verify_in_corpus,
    write_probe_results,
)
from inference.generate import GenerationResult
from inference.sampler import SamplerConfig
from model import GPT, GPTConfig
from tokenizer import Tokenizer


def test_load_probes_validates_schema(tmp_path: Path) -> None:
    """``load_probes`` rejects malformed probe files."""
    path = tmp_path / "probes.json"
    path.write_text(json.dumps([{"prompt": "Hello", "answers": ["world"]}]), encoding="utf-8")
    probes = load_probes(path)
    assert len(probes) == 1
    assert probes[0].prompt == "Hello"
    assert probes[0].answers == ("world",)

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([{"prompt": "", "answers": []}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_probes(bad_path)


def test_verify_in_corpus_filters_missing_answers() -> None:
    """Only probes with answers present in the corpus are kept."""
    probes = [
        FactualProbe(prompt="Capital of France is", answers=("Paris",)),
        FactualProbe(prompt="Capital of Atlantis is", answers=("Neptune City",)),
    ]
    verified, dropped = verify_in_corpus(probes, ["Paris is the capital of France."])
    assert len(verified) == 1
    assert verified[0].answers == ("Paris",)
    assert len(dropped) == 1


def test_completion_hits_is_case_insensitive() -> None:
    """Hit detection ignores case."""
    assert completion_hits("The answer is PARIS.", ("paris",))
    assert not completion_hits("The answer is London.", ("paris",))


def test_score_factual_probes_counts_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """``score_factual_probes`` aggregates per-probe hits into hit-rate."""
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
    probes = [
        FactualProbe(prompt="A", answers=("hit",)),
        FactualProbe(prompt="B", answers=("miss",)),
    ]

    def _fake_generate(
        _model: GPT,
        _tokenizer: Tokenizer,
        prompt: str,
        _config: object,
        **_: object,
    ) -> GenerationResult:
        text = "contains hit" if prompt == "A" else "wrong answer"
        token_ids = _tokenizer.encode(text, add_bos=False, add_eos=False)
        return GenerationResult(
            token_ids=token_ids,
            prompt_token_ids=[],
            raw_text=text,
            text=text,
        )

    import evaluation.factual_probes as factual_module

    monkeypatch.setattr(factual_module, "generate", _fake_generate)
    results = score_factual_probes(
        model,
        tokenizer,
        probes,
        sampler=SamplerConfig(temperature=0.7, top_p=0.9, repetition_penalty=1.3),
        device=torch.device("cpu"),
        max_new_tokens=8,
        seed=1,
    )
    assert results["hits"] == 1
    assert results["total"] == 2
    assert results["hit_rate"] == 0.5
    assert len(results["per_probe"]) == 2


def test_write_probe_results_round_trip(tmp_path: Path) -> None:
    """``write_probe_results`` writes readable JSON."""
    path = tmp_path / "out.json"
    payload = {"hit_rate": 0.4, "hits": 2, "total": 5, "per_probe": []}
    write_probe_results(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["hit_rate"] == 0.4
