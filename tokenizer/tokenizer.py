"""Public Tokenizer facade: train, encode, decode, save, and load."""

from __future__ import annotations

from collections.abc import Iterable

from tokenizer.dataset import sample_training_corpus
from tokenizer.decoder import decode_ids
from tokenizer.encoder import encode_text
from tokenizer.serialization import load_tokenizer, save_tokenizer
from tokenizer.trainer import train_bpe
from tokenizer.vocab import Vocab


class Tokenizer:
    """Byte-level BPE tokenizer with reserved special tokens."""

    def __init__(self, vocab: Vocab | None = None) -> None:
        """Create a tokenizer, optionally wrapping an existing vocabulary."""
        self._vocab = vocab

    @property
    def vocab(self) -> Vocab:
        """Return the underlying vocabulary."""
        if self._vocab is None:
            msg = "Tokenizer is untrained; call train() or load() first"
            raise RuntimeError(msg)
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including bytes, merges, and specials."""
        return self.vocab.vocab_size

    def train(self, corpus: Iterable[str], vocab_size: int) -> None:
        """Train byte-level BPE merges on ``corpus`` up to ``vocab_size``."""
        vocab = Vocab(vocab_size=vocab_size)
        num_merges = vocab.num_merges_allowed()
        train_bpe(vocab, corpus, num_merges)
        self._vocab = vocab

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode ``text`` to token ids."""
        return encode_text(text, self.vocab, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        """Decode token ids back to text."""
        return decode_ids(ids, self.vocab, skip_special=skip_special)

    def save(self, path: str) -> None:
        """Persist tokenizer state to ``path``."""
        save_tokenizer(path, self.vocab)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        """Load a tokenizer from ``path``."""
        vocab = load_tokenizer(path)
        return cls(vocab=vocab)

    @classmethod
    def train_sample(cls, vocab_size: int = 512) -> Tokenizer:
        """Train on the built-in sample corpus (for tests and quick sanity checks)."""
        tokenizer = cls()
        tokenizer.train(sample_training_corpus(), vocab_size=vocab_size)
        return tokenizer
