"""Download and prepare the Tiny Shakespeare corpus."""

from __future__ import annotations

import urllib.request
from pathlib import Path

DEFAULT_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_shakespeare(
    data_dir: str | Path,
    *,
    url: str = DEFAULT_SHAKESPEARE_URL,
) -> Path:
    """Download Tiny Shakespeare to ``data_dir/input.txt`` if missing."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "input.txt"
    if raw_path.exists():
        return raw_path
    with urllib.request.urlopen(url, timeout=60) as response:
        raw_path.write_bytes(response.read())
    return raw_path


def split_corpus(text: str, val_ratio: float) -> tuple[str, str]:
    """Split ``text`` into train/validation spans by character fraction."""
    if not 0.0 < val_ratio < 1.0:
        msg = f"val_ratio must be in (0, 1), got {val_ratio}"
        raise ValueError(msg)
    split_idx = int(len(text) * (1.0 - val_ratio))
    return text[:split_idx], text[split_idx:]


def prepare_shakespeare_splits(
    data_dir: str | Path,
    *,
    val_ratio: float = 0.1,
    url: str = DEFAULT_SHAKESPEARE_URL,
) -> tuple[Path, Path]:
    """Ensure train/val text files exist under ``data_dir``."""
    root = Path(data_dir)
    train_path = root / "train.txt"
    val_path = root / "val.txt"
    if train_path.exists() and val_path.exists():
        return train_path, val_path

    raw_path = download_shakespeare(root, url=url)
    text = raw_path.read_text(encoding="utf-8")
    train_text, val_text = split_corpus(text, val_ratio)
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    return train_path, val_path


def load_split_texts(data_dir: str | Path) -> tuple[str, str]:
    """Return ``(train_text, val_text)`` from prepared split files."""
    train_path, val_path = prepare_shakespeare_splits(data_dir)
    return train_path.read_text(encoding="utf-8"), val_path.read_text(encoding="utf-8")
