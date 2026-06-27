# Phase 6 — Text Generation

## Goal
A sampler supporting greedy / temperature / top-k / top-p, stop-on-`<|eos|>`, KV-cache
decoding, and a show/hide mode for the `<|think|>` reasoning span.

## Prerequisites
Phase 4 (model + KV cache), Phase 2 (tokenizer).

## Deliverables (files)
```
inference/sampler.py     # sampling strategies
inference/kvcache.py      # cache management for incremental decoding (if not in attention)
inference/generate.py     # high-level generate() + CLI
tests/test_generation.py
scripts/generate.py       # CLI entry (make generate)
```

## Spec (from Technical_Spec.md §8)
| Strategy | Params |
|---|---|
| Greedy | — |
| Temperature | `temperature` |
| Top-k | `top_k` |
| Top-p | `top_p` |
- Stop on `<|eos|>`; `max_new_tokens` guard.
- Reasoning decode modes: `show` / `hide` the `<|think|>...<|/think|>` span.
- KV-cache greedy decoding **must** match no-cache greedy decoding.

## Tasks
1. Implement logits processing for temperature, top-k, top-p (nucleus).
2. Implement incremental decoding using the KV cache from Phase 4.
3. Implement stop-on-EOS and a hard `max_new_tokens` cap (no runaway generation).
4. Implement think-span handling: parse `<|think|>...<|/think|>` and optionally strip it from
   the returned text while keeping it available.
5. Build `generate.py` + a CLI (`scripts/generate.py`).

## Tests
- Each sampling strategy returns valid tokens and respects its params.
- Greedy with KV cache == greedy without KV cache (token-for-token).
- Generation stops at `<|eos|>` and never exceeds `max_new_tokens`.
- `hide` mode removes the think span; `show` mode preserves it.

## Acceptance gate (from Expected_Outcome.md)
- `generate.py` produces coherent text and stops cleanly at EOS; each strategy works;
  KV-cache greedy matches no-cache greedy.

## Definition of done / commit
`feat: text generation (greedy/temp/top-k/top-p) with KV cache and think-span control`

## Cursor kickoff prompt
> Implement text generation per `phases/06_phase6_text_generation.md` and `Technical_Spec.md`
> §8: greedy/temperature/top-k/top-p sampling, KV-cache incremental decoding, stop-on-`<|eos|>`
> with a `max_new_tokens` cap, and show/hide handling of the `<|think|>` span. Add a CLI at
> `scripts/generate.py`. Write `tests/test_generation.py` including KV-cache vs no-cache greedy
> parity and EOS-stop behavior.
