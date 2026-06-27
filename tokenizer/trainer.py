"""Byte-level BPE merge training."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from tokenizer.vocab import Vocab

# Lossless pre-tokenization: alternating non-whitespace runs and whitespace runs.
_PRETOKENIZE_PATTERN = re.compile(r"\S+|\s+")


def pretokenize(text: str) -> list[str]:
    """Split text into chunks for byte-level BPE without losing whitespace."""
    return _PRETOKENIZE_PATTERN.findall(text)


def word_to_byte_tokens(word: tuple[int, ...]) -> list[bytes]:
    """Convert a UTF-8 byte tuple into single-byte token pieces."""
    return [bytes([byte_val]) for byte_val in word]


def get_pair_frequencies(
    splits: dict[tuple[int, ...], list[bytes]],
    word_freqs: Counter[tuple[int, ...]],
) -> Counter[tuple[bytes, bytes]]:
    """Count adjacent byte-token pair frequencies across the corpus splits."""
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    for word, frequency in word_freqs.items():
        pieces = splits[word]
        if len(pieces) < 2:
            continue
        for index in range(len(pieces) - 1):
            pair = (pieces[index], pieces[index + 1])
            pairs[pair] += frequency
    return pairs


def merge_pair_in_word(pieces: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    """Apply one merge pass to a single word's byte-token list."""
    if len(pieces) < 2:
        return pieces
    merged: list[bytes] = []
    index = 0
    while index < len(pieces):
        if (
            index < len(pieces) - 1
            and pieces[index] == pair[0]
            and pieces[index + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(pieces[index])
            index += 1
    return merged


def build_word_frequencies(corpus: Iterable[str]) -> Counter[tuple[int, ...]]:
    """Count pretokenized word byte sequences across a text corpus."""
    word_freqs: Counter[tuple[int, ...]] = Counter()
    for document in corpus:
        for chunk in pretokenize(document):
            if chunk:
                word_freqs[tuple(chunk.encode("utf-8"))] += 1
    return word_freqs


def train_bpe(vocab: Vocab, corpus: Iterable[str], num_merges: int) -> None:
    """Train ``num_merges`` byte-level BPE merges into ``vocab``."""
    if num_merges > vocab.num_merges_allowed():
        msg = (
            f"Requested {num_merges} merges but only "
            f"{vocab.num_merges_allowed()} slots are available"
        )
        raise ValueError(msg)

    word_freqs = build_word_frequencies(corpus)
    if not word_freqs:
        msg = "Corpus produced no pretokenized words"
        raise ValueError(msg)

    splits: dict[tuple[int, ...], list[bytes]] = {
        word: word_to_byte_tokens(word) for word in word_freqs
    }

    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(splits, word_freqs)
        if not pair_freqs:
            break
        best_pair, _count = pair_freqs.most_common(1)[0]
        vocab.add_merge(best_pair[0], best_pair[1])
        for word in word_freqs:
            splits[word] = merge_pair_in_word(splits[word], best_pair)
