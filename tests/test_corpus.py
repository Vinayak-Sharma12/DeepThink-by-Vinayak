"""Unit and integration tests for the LOGOS philosophy corpus (Phase 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from datasets.corpus.curated import build_bias_records, build_fallacy_records
from datasets.corpus.dedup import (
    assert_holdout_disjoint,
    deduplicate_records,
    is_near_duplicate,
    simhash,
    text_fingerprint,
)
from datasets.corpus.query import metadata_coverage, query
from datasets.corpus.sampler import CorpusSampler
from datasets.corpus.schema import CorpusRecord, load_jsonl, load_topic_registry
from datasets.corpus.split import ChunkPolicy, estimate_token_count, split_text_to_chunks

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests/fixtures/corpus_sample.jsonl"
TOPIC_REGISTRY = REPO_ROOT / "configs/topic_registry.yaml"


@pytest.fixture
def sample_records() -> list[CorpusRecord]:
    """Load the 20-row corpus fixture."""
    return load_jsonl(FIXTURE)


@pytest.fixture
def allowed_topics() -> frozenset[str]:
    """Controlled topic vocabulary."""
    return load_topic_registry(TOPIC_REGISTRY)


def test_corpus_record_validates_fixture(sample_records: list[CorpusRecord], allowed_topics: frozenset[str]) -> None:
    """Every fixture row passes CorpusRecord validation."""
    assert len(sample_records) == 20
    for record in sample_records:
        record.assert_valid(allowed_topics=allowed_topics)


def test_query_filters_by_topic_and_tradition(sample_records: list[CorpusRecord]) -> None:
    """Query API filters on topic and tradition."""
    hits = query(sample_records, topic="existence_of_god", tradition="atheism", split="train")
    assert hits
    assert all("existence_of_god" in r.topic for r in hits)
    assert all(r.tradition == "atheism" for r in hits)


def test_sampler_multi_tradition_existence_of_god(sample_records: list[CorpusRecord]) -> None:
    """sampler.sample returns chunks from at least three traditions."""
    sampler = CorpusSampler(sample_records, seed=0)
    drawn = sampler.sample("existence_of_god", 6)
    traditions = {r.tradition for r in drawn}
    assert len(traditions) >= 3


def test_sampler_buddhist_schools_separate(sample_records: list[CorpusRecord]) -> None:
    """Buddhism draws theravada, zen, tibetan as separate schools."""
    sampler = CorpusSampler(sample_records, seed=1)
    drawn = sampler.sample_buddhist_schools("no_self", 6)
    schools = {r.school for r in drawn}
    assert "theravada" in schools
    assert "zen" in schools
    assert "tibetan" in schools


def test_osho_not_tagged_atheism(sample_records: list[CorpusRecord]) -> None:
    """Osho rows use critique_of_religion tradition."""
    osho = [r for r in sample_records if r.source_id.startswith("osho_")]
    assert osho
    assert all(r.tradition == "critique_of_religion" for r in osho)


def test_prashant_vedanta_tagging(sample_records: list[CorpusRecord]) -> None:
    """Prashant uses vedanta/advaita school, not atheism/theism tradition."""
    prashant = [r for r in sample_records if r.source_id.startswith("prashant_")]
    assert prashant
    for record in prashant:
        assert record.school in {"vedanta", "advaita"}
        assert record.tradition not in {"atheism", "theism"}


def test_buddhism_not_atheism(sample_records: list[CorpusRecord]) -> None:
    """Buddhist rows carry school and buddhist tradition."""
    buddhist = [r for r in sample_records if r.tradition == "buddhist"]
    assert len(buddhist) >= 3
    assert all(r.school for r in buddhist)
    assert all(r.tradition != "atheism" for r in buddhist)


def test_dedup_exact_duplicates() -> None:
    """Exact duplicate texts are rejected."""
    base = CorpusRecord(
        id="a1",
        text="Same text chunk for dedup testing purposes here.",
        split="train",
        title="T",
        author="A",
        year=1,
        language="en",
        religion="none",
        tradition="atheism",
        topic=["logic"],
        stance="survey",
        source_type="essay",
        source_id="test",
        license="PD-US",
        token_count=10,
    )
    dup = CorpusRecord(**{**base.to_dict(), "id": "a2"})
    kept, rejected = deduplicate_records([base, dup])
    assert len(kept) == 1
    assert len(rejected) == 1


def test_simhash_near_duplicate_detection() -> None:
    """Near-identical texts exceed similarity threshold."""
    text_a = "The problem of evil challenges omnipotent theism " * 8
    text_b = text_a + "extra"
    a = simhash(text_a)
    b = simhash(text_b)
    assert is_near_duplicate(a, b, threshold=0.85)


def test_holdout_disjoint_from_train(sample_records: list[CorpusRecord]) -> None:
    """Holdout ids and fingerprints do not overlap train."""
    train = [r for r in sample_records if r.split == "train"]
    holdout = [r for r in sample_records if r.split == "holdout"]
    assert_holdout_disjoint(train, holdout)


def test_metadata_coverage_fixture(sample_records: list[CorpusRecord]) -> None:
    """Fixture train rows have full required metadata."""
    coverage = metadata_coverage(sample_records)
    for frac in coverage.values():
        assert frac >= 0.95


def test_split_text_respects_token_bounds() -> None:
    """Chunk splitter respects min and max token policy."""
    policy = ChunkPolicy(target_tokens=50, min_tokens=10, max_tokens=100)
    text = "Word " * 500
    chunks = split_text_to_chunks(text, policy)
    assert chunks
    for chunk in chunks:
        tokens = estimate_token_count(chunk)
        assert tokens >= policy.min_tokens
        assert tokens <= policy.max_tokens


def test_fallacy_corpus_minimum_rows() -> None:
    """Fallacy JSONL builder produces at least 50 rows."""
    records = build_fallacy_records()
    assert len(records) >= 50
    assert all("fallacy" in r.topic for r in records)


def test_bias_corpus_minimum_rows() -> None:
    """Bias JSONL builder produces at least 30 rows."""
    records = build_bias_records()
    assert len(records) >= 30


def test_text_fingerprint_stable() -> None:
    """Fingerprints are stable for identical normalized text."""
    text = "Hello   world\n\nTest"
    assert text_fingerprint(text) == text_fingerprint("hello world Test")


@pytest.mark.slow
def test_built_corpus_acceptance_gate() -> None:
    """Integration gate on built corpus when data/logos/train.jsonl exists."""
    train_path = REPO_ROOT / "data/logos/train.jsonl"
    if not train_path.exists():
        pytest.skip("Built corpus not present; run datasets.corpus.build_jsonl first")

    records = load_jsonl(train_path)
    holdout_path = REPO_ROOT / "data/logos/holdout/holdout.jsonl"
    holdout = load_jsonl(holdout_path) if holdout_path.exists() else []
    all_records = records + holdout

    allowed = load_topic_registry(TOPIC_REGISTRY)
    for record in records[:100]:
        record.assert_valid(allowed_topics=allowed)

    coverage = metadata_coverage(all_records)
    assert all(v >= 0.95 for v in coverage.values())

    train_tokens = sum(r.token_count for r in records)
    assert train_tokens >= 200_000

    sampler = CorpusSampler(all_records, seed=2)
    drawn = sampler.sample("existence_of_god", 6)
    assert len({r.tradition for r in drawn}) >= 3

    buddhist = sampler.sample_buddhist_schools("suffering", 6)
    schools = {r.school for r in buddhist}
    assert {"theravada", "zen", "tibetan"}.issubset(schools)

    assert_holdout_disjoint(records, holdout)

    fallacies = load_jsonl(REPO_ROOT / "data/logos/fallacies.jsonl")
    biases = load_jsonl(REPO_ROOT / "data/logos/biases.jsonl")
    assert len(fallacies) >= 50
    assert len(biases) >= 30

    audit = REPO_ROOT / "docs/corpus_audit.md"
    assert audit.exists()
