"""Near-duplicate detection and Phase 8 overlap removal."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

from datasets.corpus.schema import CorpusRecord

_WORD_RE = re.compile(r"\w+")


def text_fingerprint(text: str) -> str:
    """Exact-content fingerprint for deduplication."""
    normalized = " ".join(_WORD_RE.findall(text.lower()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def simhash(text: str, *, bits: int = 64) -> int:
    """Compute SimHash fingerprint for near-duplicate detection."""
    tokens = _WORD_RE.findall(text.lower())
    if not bits:
        return 0
    vector = [0] * bits
    for token in tokens:
        digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            bit = (digest >> i) & 1
            vector[i] += 1 if bit else -1
    fingerprint = 0
    for i, value in enumerate(vector):
        if value > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two SimHash values."""
    return (a ^ b).bit_count()


def similarity_from_simhash(a: int, b: int, *, bits: int = 64) -> float:
    """Return normalized SimHash similarity in [0, 1]."""
    if bits <= 0:
        return 0.0
    distance = hamming_distance(a, b)
    return 1.0 - distance / bits


def is_near_duplicate(
    a: int,
    b: int,
    *,
    bits: int = 64,
    threshold: float = 0.92,
) -> bool:
    """Return True if two SimHash fingerprints exceed similarity threshold."""
    return similarity_from_simhash(a, b, bits=bits) >= threshold


def deduplicate_records(
    records: Sequence[CorpusRecord],
    *,
    near_dup_threshold: float = 0.92,
    simhash_bits: int = 64,
) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
    """Remove exact and near duplicates; return (kept, rejected)."""
    kept: list[CorpusRecord] = []
    rejected: list[CorpusRecord] = []
    seen_exact: set[str] = set()
    seen_hashes: list[int] = []

    for record in records:
        exact = text_fingerprint(record.text)
        if exact in seen_exact:
            rejected.append(record)
            continue
        sh = simhash(record.text, bits=simhash_bits)
        if any(is_near_duplicate(sh, prev, bits=simhash_bits, threshold=near_dup_threshold) for prev in seen_hashes):
            rejected.append(record)
            continue
        seen_exact.add(exact)
        seen_hashes.append(sh)
        kept.append(record)
    return kept, rejected


def filter_phase8_overlap(
    records: Sequence[CorpusRecord],
    reference_texts: Iterable[str],
    *,
    threshold: float = 0.85,
    simhash_bits: int = 64,
) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
    """Reject train chunks too similar to Phase 8 general corpus."""
    ref_hashes = [simhash(text, bits=simhash_bits) for text in reference_texts]
    kept: list[CorpusRecord] = []
    rejected: list[CorpusRecord] = []
    for record in records:
        sh = simhash(record.text, bits=simhash_bits)
        if any(
            is_near_duplicate(sh, ref, bits=simhash_bits, threshold=threshold) for ref in ref_hashes
        ):
            rejected.append(record)
        else:
            kept.append(record)
    return kept, rejected


def assert_holdout_disjoint(train: Sequence[CorpusRecord], holdout: Sequence[CorpusRecord]) -> None:
    """Raise if any train/holdout id or text fingerprint overlaps."""
    train_ids = {r.id for r in train}
    holdout_ids = {r.id for r in holdout}
    overlap_ids = train_ids.intersection(holdout_ids)
    if overlap_ids:
        msg = f"holdout ids overlap train: {sorted(overlap_ids)[:5]}"
        raise ValueError(msg)

    train_fps = {text_fingerprint(r.text) for r in train}
    for record in holdout:
        if text_fingerprint(record.text) in train_fps:
            msg = f"holdout text overlaps train: {record.id}"
            raise ValueError(msg)
