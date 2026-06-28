"""Balanced sampling by topic × tradition × school for LOGOS corpus."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence

from datasets.corpus.query import query
from datasets.corpus.schema import CorpusRecord


class CorpusSampler:
    """Draw balanced multi-tradition samples from a corpus."""

    def __init__(
        self,
        records: Sequence[CorpusRecord],
        *,
        seed: int = 42,
    ) -> None:
        self._records = list(records)
        self._rng = random.Random(seed)

    @property
    def records(self) -> list[CorpusRecord]:
        """Return the underlying record list."""
        return self._records

    def sample(
        self,
        topic: str,
        n: int,
        *,
        traditions: Sequence[str] | None = None,
        schools: Sequence[str] | None = None,
        split: str = "train",
        language: str | None = None,
        stance: str | Sequence[str] | None = None,
    ) -> list[CorpusRecord]:
        """Return up to ``n`` chunks balanced across traditions and schools.

        When ``traditions`` or ``schools`` are provided, the sampler round-robins
        across those buckets so no single line dominates a contested topic.
        """
        pool = query(
            self._records,
            split=split,
            topic=topic,
            language=language,
            stance=stance,
        )
        if traditions:
            pool = [r for r in pool if r.tradition in traditions]
        if schools:
            pool = [r for r in pool if r.school in schools]

        if not pool:
            return []

        if traditions:
            buckets = _group_by(pool, lambda r: r.tradition)
            order = [t for t in traditions if t in buckets]
            return _round_robin_sample(buckets, order, n, self._rng)

        if schools:
            buckets = _group_by(pool, lambda r: r.school or "unknown")
            order = [s for s in schools if s in buckets]
            return _round_robin_sample(buckets, order, n, self._rng)

        # Default: round-robin across all traditions present in the pool.
        buckets = _group_by(pool, lambda r: r.tradition)
        order = sorted(buckets.keys())
        return _round_robin_sample(buckets, order, n, self._rng)

    def sample_buddhist_schools(
        self,
        topic: str,
        n: int,
        *,
        schools: Sequence[str] = ("theravada", "zen", "tibetan"),
        split: str = "train",
    ) -> list[CorpusRecord]:
        """Draw Buddhist chunks with separate school buckets."""
        return self.sample(topic, n, schools=schools, split=split)


def _group_by(
    records: Sequence[CorpusRecord],
    key_fn: Callable[[CorpusRecord], str],
) -> dict[str, list[CorpusRecord]]:
    buckets: dict[str, list[CorpusRecord]] = defaultdict(list)
    for record in records:
        buckets[key_fn(record)].append(record)
    return dict(buckets)


def _round_robin_sample(
    buckets: dict[str, list[CorpusRecord]],
    order: Sequence[str],
    n: int,
    rng: random.Random,
) -> list[CorpusRecord]:
    """Round-robin across buckets until ``n`` items or buckets exhausted."""
    for key in order:
        rng.shuffle(buckets[key])
    result: list[CorpusRecord] = []
    indices = {key: 0 for key in order}
    while len(result) < n:
        added = False
        for key in order:
            idx = indices[key]
            bucket = buckets.get(key, [])
            if idx < len(bucket):
                result.append(bucket[idx])
                indices[key] = idx + 1
                added = True
                if len(result) >= n:
                    break
        if not added:
            break
    return result
