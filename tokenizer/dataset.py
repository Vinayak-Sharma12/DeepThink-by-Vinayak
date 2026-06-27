"""Corpus loading utilities for BPE tokenizer training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_text_files(
    directory: Path,
    *,
    suffixes: tuple[str, ...] = (".txt", ".md"),
) -> Iterator[str]:
    """Yield file contents from ``directory`` for supported text suffixes."""
    if not directory.is_dir():
        msg = f"Not a directory: {directory}"
        raise NotADirectoryError(msg)
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path.read_text(encoding="utf-8")


def iter_corpus(paths: Iterable[Path | str]) -> Iterator[str]:
    """Yield documents from explicit file paths or directories."""
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            yield from iter_text_files(path)
        elif path.is_file():
            yield path.read_text(encoding="utf-8")
        else:
            msg = f"Path does not exist: {path}"
            raise FileNotFoundError(msg)


def sample_training_corpus() -> list[str]:
    """Small general + philosophy sample for tests and sanity-check training."""
    return [
        "The unexamined life is not worth living. — Socrates",
        "Cogito, ergo sum. Descartes sought certainty beyond doubt.",
        "What is the telos of human flourishing in Aristotle's ethics?",
        "Epoché suspends judgment; phenomenology begins in experience.",
        "Qualia are the subjective feels of conscious experience.",
        "Dharma in Hindu thought points toward duty, law, and cosmic order.",
        "Apophatic theology speaks of the divine by negation, not attribution.",
        "Logic demands valid inference; rhetoric persuades without proof.",
        "Tab\tnewline\nunicode: café, naïve, 日本語, emoji 🎭, math ∑.",
        "Reason weighs evidence; faith commits where evidence runs out.",
    ]
