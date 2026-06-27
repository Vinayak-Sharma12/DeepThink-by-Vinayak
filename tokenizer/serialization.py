"""Save and load tokenizer artifacts as versioned JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizer.vocab import SPECIAL_TOKENS, Vocab

FORMAT_VERSION = 1
ARTIFACT_FILENAME = "tokenizer.json"


@dataclass(frozen=True)
class TokenizerArtifact:
    """Serialized tokenizer state."""

    format_version: int
    vocab_size: int
    special_tokens: tuple[str, ...]
    merges: list[tuple[bytes, bytes]]


def _bytes_to_json(data: bytes) -> list[int]:
    """Encode bytes as a JSON-safe list of ints."""
    return list(data)


def _json_to_bytes(values: list[int]) -> bytes:
    """Decode a JSON int list back to bytes."""
    return bytes(values)


def artifact_from_vocab(vocab: Vocab) -> TokenizerArtifact:
    """Build a serializable artifact from an in-memory vocabulary."""
    special_tokens = tuple(
        vocab.id_to_special[token_id] for token_id in sorted(vocab.id_to_special)
    )
    return TokenizerArtifact(
        format_version=FORMAT_VERSION,
        vocab_size=vocab.vocab_size,
        special_tokens=special_tokens,
        merges=list(vocab.merges),
    )


def vocab_from_artifact(artifact: TokenizerArtifact) -> Vocab:
    """Reconstruct a vocabulary from a serialized artifact."""
    if artifact.format_version != FORMAT_VERSION:
        msg = f"Unsupported tokenizer format version: {artifact.format_version}"
        raise ValueError(msg)
    return Vocab.from_merges(
        vocab_size=artifact.vocab_size,
        merges=list(artifact.merges),
        special_tokens=artifact.special_tokens,
    )


def save_tokenizer(path: str | Path, vocab: Vocab) -> None:
    """Persist vocabulary and merges to ``path/tokenizer.json``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = artifact_from_vocab(vocab)
    payload: dict[str, Any] = {
        "format_version": artifact.format_version,
        "vocab_size": artifact.vocab_size,
        "special_tokens": list(artifact.special_tokens),
        "merges": [
            {"left": _bytes_to_json(left), "right": _bytes_to_json(right)}
            for left, right in artifact.merges
        ],
    }
    output_path = directory / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Vocab:
    """Load vocabulary and merges from ``path/tokenizer.json``."""
    input_path = Path(path) / ARTIFACT_FILENAME
    if not input_path.is_file():
        msg = f"Tokenizer artifact not found: {input_path}"
        raise FileNotFoundError(msg)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    merges = [
        (_json_to_bytes(entry["left"]), _json_to_bytes(entry["right"]))
        for entry in payload["merges"]
    ]
    special_tokens = tuple(payload.get("special_tokens", SPECIAL_TOKENS))
    artifact = TokenizerArtifact(
        format_version=int(payload["format_version"]),
        vocab_size=int(payload["vocab_size"]),
        special_tokens=special_tokens,
        merges=merges,
    )
    return vocab_from_artifact(artifact)
