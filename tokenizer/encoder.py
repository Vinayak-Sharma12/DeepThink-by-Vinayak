"""Text-to-token-id encoding for byte-level BPE."""

from __future__ import annotations

import re

from tokenizer.trainer import pretokenize
from tokenizer.vocab import SPECIAL_TOKENS, Vocab


def split_by_special_tokens(text: str, special_tokens: tuple[str, ...]) -> list[tuple[str, str]]:
    """Split ``text`` into alternating plain-text and special-token segments."""
    if not special_tokens:
        return [("text", text)]
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(token) for token in ordered)
    raw_parts = re.split(f"({pattern})", text)
    segments: list[tuple[str, str]] = []
    for part in raw_parts:
        if not part:
            continue
        if part in special_tokens:
            segments.append(("special", part))
        else:
            segments.append(("text", part))
    return segments


def bpe_encode_piece(piece: str, vocab: Vocab) -> list[int]:
    """Apply learned BPE merges to one pretokenized text chunk."""
    byte_tokens = [bytes([byte_val]) for byte_val in piece.encode("utf-8")]
    while len(byte_tokens) >= 2:
        best_rank = float("inf")
        best_pair: tuple[bytes, bytes] | None = None
        for index in range(len(byte_tokens) - 1):
            pair = (byte_tokens[index], byte_tokens[index + 1])
            rank = vocab.merge_ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = pair
        if best_pair is None:
            break
        new_tokens: list[bytes] = []
        index = 0
        while index < len(byte_tokens):
            if (
                index < len(byte_tokens) - 1
                and byte_tokens[index] == best_pair[0]
                and byte_tokens[index + 1] == best_pair[1]
            ):
                new_tokens.append(best_pair[0] + best_pair[1])
                index += 2
            else:
                new_tokens.append(byte_tokens[index])
                index += 1
        byte_tokens = new_tokens
    return [vocab.bytes_to_id[token] for token in byte_tokens]


def encode_text(
    text: str,
    vocab: Vocab,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    """Encode ``text`` to token ids, preserving special tokens as atomic ids."""
    token_ids: list[int] = []
    if add_bos:
        token_ids.append(vocab.special_token_id("<|bos|>"))

    for segment_type, content in split_by_special_tokens(text, SPECIAL_TOKENS):
        if segment_type == "special":
            token_ids.append(vocab.special_token_id(content))
            continue
        for chunk in pretokenize(content):
            token_ids.extend(bpe_encode_piece(chunk, vocab))

    if add_eos:
        token_ids.append(vocab.special_token_id("<|eos|>"))
    return token_ids
