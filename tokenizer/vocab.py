"""Vocabulary and special-token registry for byte-level BPE."""

from __future__ import annotations

from dataclasses import dataclass, field

BYTE_VOCAB_SIZE = 256

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|think|>",
    "<|/think|>",
    "<|cite|>",
    "<|/cite|>",
)


@dataclass
class Vocab:
    """Byte-level BPE vocabulary with reserved special-token ids at the end."""

    vocab_size: int
    id_to_bytes: dict[int, bytes] = field(default_factory=dict)
    bytes_to_id: dict[bytes, int] = field(default_factory=dict)
    special_to_id: dict[str, int] = field(default_factory=dict)
    id_to_special: dict[int, str] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    merge_ranks: dict[tuple[bytes, bytes], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vocab_size < BYTE_VOCAB_SIZE + len(SPECIAL_TOKENS):
            msg = (
                f"vocab_size must be at least {BYTE_VOCAB_SIZE + len(SPECIAL_TOKENS)}, "
                f"got {self.vocab_size}"
            )
            raise ValueError(msg)
        if not self.id_to_bytes:
            self._init_byte_tokens()
        if not self.special_to_id:
            self.register_special_tokens(SPECIAL_TOKENS)

    def _init_byte_tokens(self) -> None:
        """Seed the vocabulary with 256 single-byte tokens."""
        for byte_val in range(BYTE_VOCAB_SIZE):
            token = bytes([byte_val])
            self.id_to_bytes[byte_val] = token
            self.bytes_to_id[token] = byte_val

    def register_special_tokens(self, tokens: tuple[str, ...] | list[str]) -> None:
        """Assign fixed ids at the end of the vocabulary for atomic special tokens."""
        token_list = list(tokens)
        if len(token_list) > self.max_merge_slots():
            msg = f"Too many special tokens ({len(token_list)}) for vocab_size {self.vocab_size}"
            raise ValueError(msg)
        start_id = self.vocab_size - len(token_list)
        self.special_to_id.clear()
        self.id_to_special.clear()
        for offset, token in enumerate(token_list):
            token_id = start_id + offset
            self.special_to_id[token] = token_id
            self.id_to_special[token_id] = token

    def max_merge_slots(self) -> int:
        """Number of merge tokens that fit below the reserved special-token block."""
        return self.vocab_size - BYTE_VOCAB_SIZE - len(self.special_to_id)

    def num_merges_allowed(self) -> int:
        """Maximum BPE merges for this vocabulary size."""
        return self.max_merge_slots()

    def add_merge(self, left: bytes, right: bytes) -> int:
        """Append a merge pair and register the merged byte sequence."""
        if len(self.merges) >= self.max_merge_slots():
            msg = f"Cannot add merge: limit {self.max_merge_slots()} reached"
            raise ValueError(msg)
        merged = left + right
        if merged in self.bytes_to_id:
            msg = f"Merged token already exists: {merged!r}"
            raise ValueError(msg)
        new_id = BYTE_VOCAB_SIZE + len(self.merges)
        rank = len(self.merges)
        self.merges.append((left, right))
        self.merge_ranks[(left, right)] = rank
        self.id_to_bytes[new_id] = merged
        self.bytes_to_id[merged] = new_id
        return new_id

    def is_special_id(self, token_id: int) -> bool:
        """Return True when ``token_id`` is a reserved special token."""
        return token_id in self.id_to_special

    def is_special_token(self, token: str) -> bool:
        """Return True when ``token`` is a registered special string."""
        return token in self.special_to_id

    def special_token_id(self, token: str) -> int:
        """Look up the id for a special token string."""
        if token not in self.special_to_id:
            msg = f"Unknown special token: {token!r}"
            raise KeyError(msg)
        return self.special_to_id[token]

    def token_bytes(self, token_id: int) -> bytes:
        """Return raw bytes for a non-special token id."""
        if self.is_special_id(token_id):
            msg = f"Token id {token_id} is special, not byte-backed"
            raise KeyError(msg)
        return self.id_to_bytes[token_id]

    @classmethod
    def from_merges(
        cls,
        vocab_size: int,
        merges: list[tuple[bytes, bytes]],
        special_tokens: tuple[str, ...] | list[str] = SPECIAL_TOKENS,
    ) -> Vocab:
        """Rebuild a vocabulary from an ordered merge list."""
        vocab = cls(vocab_size=vocab_size)
        vocab.register_special_tokens(special_tokens)
        for left, right in merges:
            vocab.add_merge(left, right)
        return vocab
