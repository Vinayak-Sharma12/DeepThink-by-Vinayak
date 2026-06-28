"""Text cleaning for the LOGOS philosophy corpus."""

from __future__ import annotations

from datasets.cleaning import clean_text

__all__ = ["clean_text", "clean_corpus_text"]


def clean_corpus_text(text: str) -> str:
    """Normalize corpus text via the shared cleaning pipeline."""
    return clean_text(text)
