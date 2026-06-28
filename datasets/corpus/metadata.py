"""Metadata normalization and validation for corpus records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from datasets.corpus.schema import CorpusRecord, load_topic_registry


def load_source_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load source_id → metadata mapping from YAML."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    sources = payload.get("sources", {})
    return {str(k): dict(v) for k, v in sources.items()}


def apply_source_defaults(
    record: CorpusRecord,
    registry: dict[str, dict[str, Any]],
) -> CorpusRecord:
    """Merge registry defaults into a record (record fields take precedence)."""
    defaults = registry.get(record.source_id, {})
    data = record.to_dict()
    for key, value in defaults.items():
        if key not in data or data[key] in (None, "", []):
            data[key] = value
    return CorpusRecord.from_dict(data)


def normalize_record(record: CorpusRecord) -> CorpusRecord:
    """Normalize string fields and topic lists."""
    data = record.to_dict()
    data["topic"] = sorted({t.strip() for t in data.get("topic", []) if t.strip()})
    data["tradition"] = str(data.get("tradition", "")).strip()
    data["religion"] = str(data.get("religion", "")).strip()
    if data.get("school"):
        data["school"] = str(data["school"]).strip()
    return CorpusRecord.from_dict(data)


def validate_records(
    records: Sequence[CorpusRecord],
    *,
    topic_registry_path: str | Path,
) -> list[tuple[CorpusRecord, list[str]]]:
    """Validate all records; return (record, errors) pairs with errors."""
    allowed = load_topic_registry(topic_registry_path)
    results: list[tuple[CorpusRecord, list[str]]] = []
    for record in records:
        normalized = normalize_record(record)
        errors = normalized.validate(allowed_topics=allowed)
        results.append((normalized, errors))
    return results


def filter_valid_records(
    records: Sequence[CorpusRecord],
    *,
    topic_registry_path: str | Path,
) -> tuple[list[CorpusRecord], list[tuple[CorpusRecord, list[str]]]]:
    """Return valid records and invalid (record, errors) pairs."""
    checked = validate_records(records, topic_registry_path=topic_registry_path)
    valid: list[CorpusRecord] = []
    invalid: list[tuple[CorpusRecord, list[str]]] = []
    for record, errors in checked:
        if errors:
            invalid.append((record, errors))
        else:
            valid.append(record)
    return valid, invalid
