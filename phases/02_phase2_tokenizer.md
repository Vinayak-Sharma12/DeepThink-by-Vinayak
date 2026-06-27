# Phase 2 — Tokenizer (BPE from scratch)

## Goal
Implement byte-level Byte Pair Encoding from scratch with lossless round-trip and all
reserved special tokens, trained on the combined (general + philosophy) corpus.

## Prerequisites
Phase 1 (scaffold + tests run).

## Deliverables (files)
```
tokenizer/dataset.py        # corpus loading/iteration for training
tokenizer/vocab.py          # vocab + special-token registry
tokenizer/trainer.py        # BPE merge training
tokenizer/encoder.py        # text -> ids
tokenizer/decoder.py        # ids -> text
tokenizer/serialization.py  # save/load vocab + merges
tokenizer/tokenizer.py      # public Tokenizer facade (train/encode/decode/save/load)
tests/test_tokenizer.py
```

## Spec (from Technical_Spec.md §3)
- Byte-level BPE; lossless `decode(encode(x)) == x` for arbitrary unicode/whitespace.
- Target `vocab_size = 32_000` (configurable).
- Reserved special tokens (never split by merges):
  `<|bos|> <|eos|> <|pad|> <|unk|> <|system|> <|user|> <|assistant|> <|think|> <|/think|> <|cite|> <|/cite|>`

## Public interface
```python
class Tokenizer:
    def train(self, corpus: Iterable[str], vocab_size: int) -> None: ...
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], *, skip_special: bool = True) -> str: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Tokenizer": ...
```

## Tasks
1. Implement byte-level pre-tokenization (work on raw bytes so any input round-trips).
2. Implement BPE training: count pair frequencies, merge most frequent, repeat to vocab size.
3. Register special tokens as atomic ids; ensure the merge table can never split them.
4. Implement encode (apply merges greedily) and decode (invert, bytes → utf-8).
5. Implement serialization (JSON for vocab + merges; versioned).
6. Train on a small combined sample and check the compression ratio is reasonable.

## Tests
- Lossless round-trip on held-out text incl. emoji, accents, tabs/newlines, mixed scripts.
- Domain terms (*epoché, qualia, dharma, telos*) do **not** explode into single bytes.
- Each special token encodes to exactly one id and survives round-trip.
- Save → load → identical encode/decode behavior.
- Merge-operation unit tests (a known toy corpus produces the expected merges).

## Acceptance gate (from Expected_Outcome.md)
- Lossless round-trip on held-out text; reasonable compression ratio; special tokens intact.

## Definition of done / commit
`feat: from-scratch byte-level BPE tokenizer with reserved special tokens`

## Cursor kickoff prompt
> Implement a from-scratch byte-level BPE tokenizer per `phases/02_phase2_tokenizer.md` and
> `Technical_Spec.md` §3, with the `Tokenizer` facade shown there. Register all reserved
> special tokens as atomic ids that merges can never split. Write `tests/test_tokenizer.py`
> proving lossless unicode/whitespace round-trip, special-token integrity, save/load parity,
> and correct merges on a toy corpus. No external tokenizer libraries.
