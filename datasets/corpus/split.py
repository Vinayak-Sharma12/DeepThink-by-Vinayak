"""Split long texts into token-bounded corpus chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets.corpus.clean import clean_corpus_text

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ChunkPolicy:
    """Token bounds for corpus chunking."""

    target_tokens: int = 512
    min_tokens: int = 64
    max_tokens: int = 1024
    overlap_tokens: int = 0


def estimate_token_count(text: str) -> int:
    """Estimate BPE-like token count (~0.75 words per token for English prose)."""
    words = len(_WORD_RE.findall(text))
    return max(1, int(words * 1.33))


def split_text_to_chunks(
    text: str,
    policy: ChunkPolicy,
) -> list[str]:
    """Split cleaned text into chunks respecting token bounds."""
    cleaned = clean_corpus_text(text)
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = "\n\n".join(current)
        if estimate_token_count(body) >= policy.min_tokens:
            chunks.append(body)
        current = []
        current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_token_count(para)
        if para_tokens > policy.max_tokens:
            flush()
            chunks.extend(_split_oversized_paragraph(para, policy))
            continue
        if current_tokens + para_tokens > policy.target_tokens and current:
            flush()
        current.append(para)
        current_tokens += para_tokens
        if current_tokens >= policy.target_tokens:
            flush()

    flush()
    return chunks


def _split_oversized_paragraph(text: str, policy: ChunkPolicy) -> list[str]:
    """Split a long paragraph by sentences or words when it exceeds max_tokens."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return _split_by_words(text, policy)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_tokens = estimate_token_count(sentence)
        if sent_tokens > policy.max_tokens:
            if current:
                body = " ".join(current)
                if estimate_token_count(body) >= policy.min_tokens:
                    chunks.append(body)
                current = []
                current_tokens = 0
            chunks.extend(_split_by_words(sentence, policy))
            continue
        if current_tokens + sent_tokens > policy.target_tokens and current:
            body = " ".join(current)
            if estimate_token_count(body) >= policy.min_tokens:
                chunks.append(body)
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        body = " ".join(current)
        if estimate_token_count(body) >= policy.min_tokens:
            chunks.append(body)
    return chunks


def _split_by_words(text: str, policy: ChunkPolicy) -> list[str]:
    """Hard-split by word count when sentences are still too long."""
    words = text.split()
    if not words:
        return []
    words_per_chunk = max(policy.target_tokens, 1)
    chunks: list[str] = []
    for start in range(0, len(words), words_per_chunk):
        chunk_words = words[start : start + words_per_chunk]
        body = " ".join(chunk_words)
        if estimate_token_count(body) >= policy.min_tokens:
            chunks.append(body)
    return chunks
