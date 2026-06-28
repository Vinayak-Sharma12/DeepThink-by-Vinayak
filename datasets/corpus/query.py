"""Filter and query API over LOGOS corpus metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from datasets.corpus.schema import CorpusRecord


def query(
    records: Sequence[CorpusRecord],
    *,
    split: str | None = None,
    tradition: str | Sequence[str] | None = None,
    school: str | Sequence[str] | None = None,
    religion: str | Sequence[str] | None = None,
    stance: str | Sequence[str] | None = None,
    source_type: str | Sequence[str] | None = None,
    source_id: str | Sequence[str] | None = None,
    license: str | Sequence[str] | None = None,
    language: str | None = None,
    topic: str | Sequence[str] | None = None,
    topic_mode: str = "any",
    mix_bucket: str | Sequence[str] | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    predicate: Callable[[CorpusRecord], bool] | None = None,
) -> list[CorpusRecord]:
    """Filter corpus records by metadata fields.

  ``topic_mode`` is ``any`` (intersection non-empty) or ``all`` (superset).
    """
    result: list[CorpusRecord] = []
    for record in records:
        if split is not None and record.split != split:
            continue
        if not _match_scalar_or_seq(record.tradition, tradition):
            continue
        if not _match_scalar_or_seq(record.school, school):
            continue
        if not _match_scalar_or_seq(record.religion, religion):
            continue
        if not _match_scalar_or_seq(record.stance, stance):
            continue
        if not _match_scalar_or_seq(record.source_type, source_type):
            continue
        if not _match_scalar_or_seq(record.source_id, source_id):
            continue
        if not _match_scalar_or_seq(record.license, license):
            continue
        if language is not None and record.language != language:
            continue
        if mix_bucket is not None and not _match_scalar_or_seq(record.mix_bucket, mix_bucket):
            continue
        if min_tokens is not None and record.token_count < min_tokens:
            continue
        if max_tokens is not None and record.token_count > max_tokens:
            continue
        if topic is not None and not _match_topic(record.topic, topic, mode=topic_mode):
            continue
        if predicate is not None and not predicate(record):
            continue
        result.append(record)
    return result


def count_by_field(
    records: Iterable[CorpusRecord],
    field_name: str,
    *,
    split: str | None = "train",
) -> dict[str, int]:
    """Count records grouped by a metadata field."""
    counts: dict[str, int] = {}
    for record in records:
        if split is not None and record.split != split:
            continue
        key = str(getattr(record, field_name) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def topic_tradition_heatmap(
    records: Iterable[CorpusRecord],
    *,
    split: str | None = "train",
) -> dict[str, dict[str, int]]:
    """Build topic × tradition counts for audit reports."""
    heatmap: dict[str, dict[str, int]] = {}
    for record in records:
        if split is not None and record.split != split:
            continue
        for topic in record.topic:
            bucket = heatmap.setdefault(topic, {})
            trad = record.tradition or "unknown"
            bucket[trad] = bucket.get(trad, 0) + 1
    return heatmap


def metadata_coverage(
    records: Sequence[CorpusRecord],
    *,
    split: str = "train",
) -> dict[str, float]:
    """Fraction of train rows with required metadata populated."""
    subset = [r for r in records if r.split == split]
    if not subset:
        return {"stance": 0.0, "tradition": 0.0, "topic": 0.0, "source_type": 0.0, "license": 0.0}
    fields = ("stance", "tradition", "topic", "source_type", "license")
    coverage: dict[str, float] = {}
    for field_name in fields:
        filled = sum(1 for r in subset if _field_populated(r, field_name))
        coverage[field_name] = filled / len(subset)
    return coverage


def _field_populated(record: CorpusRecord, field_name: str) -> bool:
    value: Any = getattr(record, field_name)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def _match_scalar_or_seq(value: str | None, pattern: str | Sequence[str] | None) -> bool:
    if pattern is None:
        return True
    if value is None:
        return False
    if isinstance(pattern, str):
        return value == pattern
    return value in pattern


def _match_topic(
    record_topics: list[str],
    query_topic: str | Sequence[str],
    *,
    mode: str,
) -> bool:
    if isinstance(query_topic, str):
        topics = {query_topic}
    else:
        topics = set(query_topic)
    record_set = set(record_topics)
    if mode == "all":
        return topics.issubset(record_set)
    return bool(record_set.intersection(topics))
