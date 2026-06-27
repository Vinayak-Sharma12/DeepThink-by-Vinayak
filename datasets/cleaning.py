"""Text normalization and cleaning for the dataset pipeline."""

from __future__ import annotations

import re
import unicodedata

# Horizontal whitespace runs (spaces/tabs) collapse to a single space; newlines are kept.
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


def clean_text(text: str) -> str:
    """Normalize unicode and remove harmful control characters.

    Cleaning is conservative so tokenizer round-trip behavior stays intact:
    newlines and tabs are preserved, only non-printable control characters are
    stripped, and repeated horizontal whitespace is collapsed.
    """
    normalized = unicodedata.normalize("NFC", text)
    kept_chars: list[str] = []
    for char in normalized:
        if char in {"\n", "\t"}:
            kept_chars.append(char)
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        kept_chars.append(char)
    collapsed = _HORIZONTAL_WHITESPACE.sub(" ", "".join(kept_chars))
    return collapsed.strip()
