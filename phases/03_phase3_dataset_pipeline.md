# Phase 3 — Dataset Pipeline

## Goal
Turn raw text into training-ready batches: clean → chunk → tokenize → pack → loss-mask →
collate. Loss masking is the load-bearing piece for all later SFT/persona/DPO phases.

## Prerequisites
Phase 2 (tokenizer).

## Deliverables (files)
```
datasets/cleaning.py     # text normalization / cleaning
datasets/chunking.py     # split long docs into ctx-length windows
datasets/packing.py      # pack token streams to ctx_len
datasets/masking.py      # loss-mask utility (prompt masked, completion counted)
datasets/dataset.py      # Dataset classes (pretrain + instruction/SFT)
datasets/collate.py      # batch collator -> (input_ids, targets, loss_mask)
tests/test_dataset.py
```

## Spec (from Technical_Spec.md §5)
Batch contract:
| Tensor | Shape | Meaning |
|---|---|---|
| `input_ids` | `[B, T]` | token ids |
| `targets` | `[B, T]` | `input_ids` shifted by 1 |
| `loss_mask` | `[B, T]` | 1 where loss counts, 0 elsewhere |

- Pretrain dataset: contiguous packed tokens, loss on all positions.
- SFT/instruction dataset: role-formatted (`<|system|><|user|><|assistant|>`); **prompt
  tokens masked (loss = 0), only the assistant span contributes to loss.**
- Deterministic sampling under a fixed seed.

## Tasks
1. Implement cleaning (unicode normalize, strip control chars, collapse whitespace — but
   keep round-trip safe; do not corrupt the tokenizer's assumptions).
2. Implement chunking + packing to `ctx_len` with correct `<|bos|>`/`<|eos|>` placement.
3. Implement `build_loss_mask(prompt_len, total_len)` and the SFT formatter that produces
   `input_ids`, `targets` (shift by 1), and `loss_mask`.
4. Implement `Dataset` classes and a `collate_fn` that pads to the batch max and sets pad
   positions' `loss_mask = 0`.
5. Make sampling deterministic given a seed.

## Tests
- `loss_mask` is 0 across the entire prompt span and 1 across the completion span.
- `targets[t] == input_ids[t+1]` (no off-by-one); last-position handling correct.
- Padding positions never contribute to loss.
- Fixed seed → identical batches across runs.

## Acceptance gate (from Expected_Outcome.md)
- A batch yields correct `(input_ids, targets, loss_mask)` shapes; the unit test proves the
  mask zeros prompt positions; deterministic sampling with a fixed seed.

## Definition of done / commit
`feat: dataset pipeline with from-scratch loss masking and deterministic batching`

## Cursor kickoff prompt
> Implement the dataset pipeline per `phases/03_phase3_dataset_pipeline.md` and
> `Technical_Spec.md` §5. Build cleaning, chunking, packing, the loss-mask utility, pretrain
> and SFT Dataset classes, and a collator returning `(input_ids, targets, loss_mask)`. Write
> `tests/test_dataset.py` proving the loss mask zeroes the prompt span, no input/target
> off-by-one, padding excluded from loss, and deterministic batches under a fixed seed.
