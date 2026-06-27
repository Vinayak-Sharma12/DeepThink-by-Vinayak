# Phase 7 — Tiny GPT (Shakespeare)

## Goal
Train the ~10M-param **Tiny** model on Tiny Shakespeare end-to-end — the first real "it
works" milestone proving the whole stack (tokenizer → data → model → trainer → generation).

## Prerequisites
Phases 2–6 all green (especially overfit-one-batch and reproducible resume).

## Deliverables (files)
```
configs/tiny.yaml              # Tiny config (n_layer 6, n_head 6, d_model 384, ctx 512)
data/shakespeare/...            # downloaded/prepared corpus
scripts/train.py                # generic training entry (make train CONFIG=...)
experiments/experiment_00X/     # auto-generated run record (config, loss.csv, samples)
```

## Spec
- Tiny tier: ~10M params (per `Technical_Spec.md` §4.4).
- Train tokenizer (or reuse a small vocab) on the Shakespeare text.
- Use the Phase 5 trainer; log loss; sample periodically.

## Tasks
1. Prepare the Tiny Shakespeare dataset via the Phase 3 pipeline.
2. Write `configs/tiny.yaml` and a generic `scripts/train.py` that loads a YAML config.
3. Train to the validation target; save checkpoints; record a loss curve.
4. Generate samples at several temperatures; save to the experiment folder.

## Tests / checks
- Validation perplexity reaches the target for a Tiny model on this corpus.
- Generated samples are clearly English / Shakespeare-style, not gibberish.
- Not memorizing: sampled text is not verbatim training text (spot-check).

## Acceptance gate (from Expected_Outcome.md)
- Model generates fluent Shakespeare-style English; validation perplexity hits target;
  samples are clearly English (no memorization).

## Definition of done / commit
`feat: train Tiny GPT on Shakespeare; first end-to-end run + samples`

## Cursor kickoff prompt
> Using the existing stack, train the Tiny (~10M) model on Tiny Shakespeare per
> `phases/07_phase7_tiny_gpt_shakespeare.md`. Create `configs/tiny.yaml`, a generic
> `scripts/train.py` that consumes a YAML config, run training to the validation target, and
> save a loss curve + temperature-varied samples into an `experiments/` run folder. Report
> perplexity and paste a few samples.
