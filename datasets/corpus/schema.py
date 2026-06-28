"""CorpusRecord schema and validation for the LOGOS philosophy corpus."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

VALID_STANCES = frozenset(
    {"argues_for", "argues_against", "survey", "primary_source", "commentary"}
)
VALID_SOURCE_TYPES = frozenset(
    {
        "book",
        "scripture",
        "debate_transcript",
        "dialogue_qa",
        "interview",
        "essay",
    }
)
VALID_LICENSES = frozenset({"PD-US", "licensed", "unknown"})
VALID_SPLITS = frozenset({"train", "holdout"})


class Stance(str, Enum):
    """What a corpus chunk does rhetorically."""

    ARGUES_FOR = "argues_for"
    ARGUES_AGAINST = "argues_against"
    SURVEY = "survey"
    PRIMARY_SOURCE = "primary_source"
    COMMENTARY = "commentary"


class SourceType(str, Enum):
    """Origin format of a corpus chunk."""

    BOOK = "book"
    SCRIPTURE = "scripture"
    DEBATE_TRANSCRIPT = "debate_transcript"
    DIALOGUE_QA = "dialogue_qa"
    INTERVIEW = "interview"
    ESSAY = "essay"


@dataclass
class CorpusRecord:
    """A single JSONL row in the LOGOS philosophy corpus."""

    id: str
    text: str
    split: str
    title: str
    author: str
    year: int
    language: str
    religion: str
    tradition: str
    topic: list[str]
    stance: str
    source_type: str
    source_id: str
    license: str
    token_count: int
    school: str | None = None
    translation: str | None = None
    edition: str | None = None
    source_url: str | None = None
    editor: str | None = None
    moderator: str | None = None
    debate_round: str | None = None
    chunk_index: int | None = None
    rhetorical_move: str | None = None
    paired_topics: list[str] = field(default_factory=list)
    original_language: str | None = None
    mix_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSONL-compatible dict (omit null optionals)."""
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusRecord:
        """Deserialize from a JSONL dict."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in data.items() if key in known}
        if "topic" not in filtered:
            filtered["topic"] = []
        if "paired_topics" not in filtered:
            filtered["paired_topics"] = []
        return cls(**filtered)

    def validate(self, *, allowed_topics: frozenset[str] | None = None) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if not self.id:
            errors.append("id is required")
        if not self.text.strip():
            errors.append("text must be non-empty")
        if self.split not in VALID_SPLITS:
            errors.append(f"invalid split: {self.split}")
        if self.stance not in VALID_STANCES:
            errors.append(f"invalid stance: {self.stance}")
        if self.source_type not in VALID_SOURCE_TYPES:
            errors.append(f"invalid source_type: {self.source_type}")
        if self.license not in VALID_LICENSES:
            errors.append(f"invalid license: {self.license}")
        if not self.topic:
            errors.append("topic must be non-empty")
        if allowed_topics is not None:
            unknown = [t for t in self.topic if t not in allowed_topics]
            if unknown:
                errors.append(f"unknown topics: {unknown}")
        if self.token_count < 1:
            errors.append("token_count must be positive")
        errors.extend(_validate_tagging_rules(self))
        return errors

    def assert_valid(self, *, allowed_topics: frozenset[str] | None = None) -> None:
        """Raise ValueError if the record fails validation."""
        errors = self.validate(allowed_topics=allowed_topics)
        if errors:
            msg = "; ".join(errors)
            raise ValueError(msg)


def _validate_tagging_rules(record: CorpusRecord) -> list[str]:
    """Enforce Osho, Prashant, and Buddhism tagging rules."""
    errors: list[str] = []
    source = record.source_id.lower()
    if source.startswith("osho_") and record.tradition == "atheism":
        errors.append("Osho must use tradition critique_of_religion, not atheism")
    if source.startswith("prashant_"):
        if record.school not in {"vedanta", "advaita"}:
            errors.append("Prashant must use school vedanta/advaita")
        if record.tradition in {"atheism", "theism"}:
            errors.append("Prashant must not use tradition atheism/theism")
    if record.tradition == "buddhist" or record.religion == "buddhist":
        if not record.school:
            errors.append("Buddhist rows require school (theravada|zen|tibetan|...)")
        if record.tradition == "atheism":
            errors.append("Buddhism must not be tagged as atheism tradition")
    return errors


def load_topic_registry(path: str | Path) -> frozenset[str]:
    """Load controlled topic vocabulary from YAML."""
    registry_path = Path(path)
    with registry_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    topics = payload.get("topics", [])
    return frozenset(str(t) for t in topics)


def make_record_id(text: str, source_id: str, chunk_index: int = 0) -> str:
    """Derive a stable id from content and source metadata."""
    digest = hashlib.sha256(f"{source_id}:{chunk_index}:{text[:200]}".encode()).hexdigest()
    return digest[:16]


def new_uuid_id() -> str:
    """Return a random uuid prefix for curated rows."""
    return uuid.uuid4().hex[:16]


def load_jsonl(path: str | Path) -> list[CorpusRecord]:
    """Load corpus records from a JSONL file."""
    records: list[CorpusRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(CorpusRecord.from_dict(json.loads(line)))
    return records


def write_jsonl(path: str | Path, records: list[CorpusRecord]) -> None:
    """Write corpus records to a JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
