"""Ingest debate and interview transcripts with speaker attribution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from datasets.corpus.clean import clean_corpus_text
from datasets.corpus.metadata import apply_source_defaults, load_source_registry
from datasets.corpus.schema import CorpusRecord, make_record_id
from datasets.corpus.split import ChunkPolicy, estimate_token_count, split_text_to_chunks

_ROUND_MARKERS = {
    "opening": re.compile(r"OPENING|INTRODUCTION", re.I),
    "rebuttal": re.compile(r"REBUTTAL|COUNTER", re.I),
    "cross_exam": re.compile(r"CROSS.?EXAM|QUESTION AND ANSWER|Q&A", re.I),
    "closing": re.compile(r"CLOSING", re.I),
    "audience_qa": re.compile(r"AUDIENCE", re.I),
}

_SPEAKER_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    "debate_akhtar_nadwi": [
        (r"JAVED AKHTAR", "Javed Akhtar", "atheism", "argues_against"),
        (r"MUFTI SHAMAIL NADWI", "Mufti Shamail Nadwi", "islamic", "argues_for"),
        (r"SAURABH DWIVEDI|MODERATOR", "Saurabh Dwivedi", "none", "survey"),
    ],
    "prashant_debate_critique": [
        (r"ACHARYA PRASHANT|PRASHANT", "Acharya Prashant", "hindu", "argues_against"),
    ],
    "oconnor_bigthink": [
        (r"ALEX O'CONNOR|COSMIC SKEPTIC", "Alex O'Connor", "atheism", "survey"),
    ],
}


def parse_transcript_sections(text: str) -> list[tuple[str, str, str | None]]:
    """Split transcript into (section_header, body, debate_round) tuples."""
    sections: list[tuple[str, str, str | None]] = []
    parts = re.split(r"---+([^-]+)---+", text)
    if len(parts) == 1:
        return [("", text.strip(), None)]

    if parts[0].strip():
        sections.append(("", parts[0].strip(), _detect_round(parts[0])))

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections.append((header, body, _detect_round(header)))
    return sections


def _detect_round(header: str) -> str | None:
    for round_name, pattern in _ROUND_MARKERS.items():
        if pattern.search(header):
            return round_name
    return None


def detect_speaker(
    header: str,
    body: str,
    source_id: str,
) -> tuple[str, str, str] | None:
    """Return (author, tradition, stance) for a transcript section."""
    patterns = _SPEAKER_PATTERNS.get(source_id, [])
    combined = f"{header}\n{body[:500]}"
    for pattern, author, tradition, stance in patterns:
        if re.search(pattern, combined, re.I):
            return author, tradition, stance
    return None


def ingest_transcript(
    source_id: str,
    path: Path,
    registry: dict[str, dict[str, Any]],
    policy: ChunkPolicy,
) -> list[CorpusRecord]:
    """Ingest a debate/interview transcript with speaker tags."""
    meta = registry.get(source_id, {})
    if not meta:
        return []

    raw = path.read_text(encoding="utf-8", errors="replace")
    sections = parse_transcript_sections(raw)
    records: list[CorpusRecord] = []

    for sec_idx, (header, body, debate_round) in enumerate(sections):
        speaker = detect_speaker(header, body, source_id)
        author = speaker[0] if speaker else str(meta.get("author", "Unknown"))
        tradition = speaker[1] if speaker else str(meta.get("tradition", "none"))
        stance = speaker[2] if speaker else str(meta.get("stance", "survey"))

        chunks = split_text_to_chunks(body, policy)
        for chunk_idx, chunk in enumerate(chunks):
            global_idx = sec_idx * 1000 + chunk_idx
            record = CorpusRecord(
                id=make_record_id(chunk, source_id, global_idx),
                text=chunk,
                split="train",
                title=str(meta.get("title", source_id)),
                author=author,
                year=int(meta.get("year", 0)),
                language=str(meta.get("language", "en")),
                religion=str(meta.get("religion", "none")),
                tradition=tradition,
                school=meta.get("school"),
                topic=list(meta.get("topic", ["existence_of_god"])),
                stance=stance,
                source_type=str(meta.get("source_type", "debate_transcript")),
                source_id=source_id,
                license=str(meta.get("license", "licensed")),
                token_count=estimate_token_count(chunk),
                moderator="Saurabh Dwivedi" if source_id == "debate_akhtar_nadwi" and speaker and speaker[2] == "survey" else None,
                debate_round=debate_round,
                original_language=meta.get("original_language"),
                mix_bucket=meta.get("mix_bucket"),
                chunk_index=global_idx,
            )
            records.append(apply_source_defaults(record, registry))

    return records


def ingest_all_transcripts(
    repo_root: Path,
    registry_path: Path,
    policy: ChunkPolicy,
) -> list[CorpusRecord]:
    """Ingest all transcript sources from the registry."""
    registry = load_source_registry(registry_path)
    records: list[CorpusRecord] = []
    for source_id, meta in registry.items():
        if meta.get("source_type") not in {"debate_transcript", "dialogue_qa", "interview"}:
            continue
        rel = meta.get("file")
        if not rel:
            continue
        path = repo_root / rel
        if not path.exists():
            continue
        records.extend(ingest_transcript(source_id, path, registry, policy))
    return records
