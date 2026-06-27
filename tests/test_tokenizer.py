"""Unit and integration tests for the byte-level BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.dataset import sample_training_corpus
from tokenizer.tokenizer import Tokenizer
from tokenizer.trainer import build_word_frequencies, train_bpe
from tokenizer.vocab import SPECIAL_TOKENS, Vocab

HELD_OUT_TEXTS = [
    "Plain ASCII with tabs\there and newlines\nthere.",
    "Unicode accents: café, naïve, Zürich, epoché, apophatic.",
    "Mixed scripts: English, 日本語, Ελληνικά, العربية.",
    "Emoji and symbols: 🎭🔥 ∑∞ ≠ ≤",
    "  leading and trailing spaces  ",
    "Philosophy: qualia, dharma, telos, epoché, apophatic theology.",
]

DOMAIN_TERMS = ("epoché", "qualia", "dharma", "telos", "apophatic")

TOY_CORPUS = ["aa", "aaa", "aab", "aba"]


@pytest.fixture
def trained_tokenizer(tmp_path: Path) -> Tokenizer:
    """Tokenizer trained on the sample corpus."""
    tokenizer = Tokenizer()
    tokenizer.train(sample_training_corpus(), vocab_size=512)
    return tokenizer


def test_lossless_round_trip_held_out(trained_tokenizer: Tokenizer) -> None:
    """decode(encode(x)) == x for diverse held-out unicode and whitespace."""
    for text in HELD_OUT_TEXTS:
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text, repr(text)


def test_domain_terms_compress(trained_tokenizer: Tokenizer) -> None:
    """Domain terms should not remain one id per raw byte after training."""
    for term in DOMAIN_TERMS:
        token_ids = trained_tokenizer.encode(term)
        byte_len = len(term.encode("utf-8"))
        assert len(token_ids) < byte_len, term


@pytest.mark.parametrize("special", SPECIAL_TOKENS)
def test_special_token_single_id(trained_tokenizer: Tokenizer, special: str) -> None:
    """Each reserved special token encodes to exactly one id."""
    token_ids = trained_tokenizer.encode(special)
    assert len(token_ids) == 1
    assert token_ids[0] == trained_tokenizer.vocab.special_token_id(special)


def test_special_token_round_trip(trained_tokenizer: Tokenizer) -> None:
    """Special tokens survive encode/decode when skip_special=False."""
    text = "Hello <|user|> world <|assistant|>!"
    token_ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(token_ids, skip_special=False)
    assert decoded == text


def test_bos_eos_flags(trained_tokenizer: Tokenizer) -> None:
    """add_bos/add_eos insert atomic special ids without altering body text."""
    body = "VERITAS commits."
    token_ids = trained_tokenizer.encode(body, add_bos=True, add_eos=True)
    assert token_ids[0] == trained_tokenizer.vocab.special_token_id("<|bos|>")
    assert token_ids[-1] == trained_tokenizer.vocab.special_token_id("<|eos|>")
    assert trained_tokenizer.decode(token_ids[1:-1]) == body


def test_save_load_parity(trained_tokenizer: Tokenizer, tmp_path: Path) -> None:
    """Save → load preserves encode/decode behavior."""
    save_dir = tmp_path / "tokenizer"
    trained_tokenizer.save(str(save_dir))
    loaded = Tokenizer.load(str(save_dir))
    sample = HELD_OUT_TEXTS[1]
    assert loaded.encode(sample) == trained_tokenizer.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_toy_corpus_first_merge() -> None:
    """Known toy corpus produces the expected first merge (b'a' + b'a')."""
    vocab = Vocab(vocab_size=280)
    train_bpe(vocab, TOY_CORPUS, num_merges=1)
    assert vocab.merges == [(b"a", b"a")]


def test_toy_corpus_merge_application() -> None:
    """After training on the toy corpus, 'aa' becomes a single merged token."""
    vocab = Vocab(vocab_size=280)
    train_bpe(vocab, TOY_CORPUS, num_merges=4)
    tokenizer = Tokenizer(vocab=vocab)
    token_ids = tokenizer.encode("aa")
    assert len(token_ids) == 1
    assert token_ids[0] == vocab.bytes_to_id[b"aa"]


def test_build_word_frequencies_counts_pretokenized_words() -> None:
    """Word frequency builder splits on whitespace boundaries."""
    freqs = build_word_frequencies(["aa aaa", "aa"])
    assert freqs[tuple(b"aa")] == 2
    assert freqs[tuple(b"aaa")] == 1


def test_compression_ratio_reasonable(trained_tokenizer: Tokenizer) -> None:
    """Sample corpus encoding compresses below raw UTF-8 byte length."""
    joined = "\n".join(sample_training_corpus())
    token_ids = trained_tokenizer.encode(joined)
    raw_bytes = len(joined.encode("utf-8"))
    assert len(token_ids) < raw_bytes * 0.85


def test_special_tokens_not_in_merge_vocab() -> None:
    """Reserved special strings live only in the special registry, not merge vocab."""
    tokenizer = Tokenizer()
    tokenizer.train(sample_training_corpus(), vocab_size=512)
    for special in SPECIAL_TOKENS:
        assert special.encode("utf-8") not in tokenizer.vocab.bytes_to_id
        embedded = f"prefix {special} suffix"
        token_ids = tokenizer.encode(embedded)
        decoded = tokenizer.decode(token_ids, skip_special=False)
        assert decoded == embedded
