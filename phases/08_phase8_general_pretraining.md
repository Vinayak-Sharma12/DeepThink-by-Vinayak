# Phase 8 — General Pretraining (Small)

## Goal
Pretrain the 30–50M **Small** model on a general corpus (TinyStories + WikiText + a subset of
OpenWebText) to get fluency **and** world knowledge — the factual ground the model must later
be "truthful about."

## Prerequisites
Phase 7 (the pipeline works at scale); reproducible checkpoints.

## Deliverables (files)
```
configs/small.yaml             # Small config (n_layer 8, n_head 8, d_model 512, ctx 1024)
datasets/sources/general.py     # loaders for TinyStories / WikiText / OpenWebText subset
data/general/...                # prepared shards
experiments/experiment_00X/     # run record
```

## Spec
- Small tier ~30–50M params, `ctx_len = 1024`.
- **Weight the factual/expository slice (WikiText) heavily** — fluency without facts produces
  a model that is confidently empty.
- Curriculum-friendly data ordering optional; mixed shards fine.

## Tasks
1. Implement loaders for each source and a mixing/weighting config (WikiText weighted up).
2. Prepare packed shards at `ctx_len = 1024`.
3. Train the Small model; this is **compute-bound** — checkpoint often and plan to build
   Phase 9 in parallel while it trains.
4. Track perplexity on a general held-out set.

## Tests / checks
- Target perplexity on a general held-out set.
- Coherent general-purpose completions; basic world knowledge present.
- The expository slice is well represented (sanity-check a few factual prompts).

## Acceptance gate (from Expected_Outcome.md)
- Coherent general completions + basic world knowledge; target perplexity on held-out set;
  factual/expository slice well represented (not fluent-but-empty).

## Definition of done / commit
`feat: general pretraining of the Small model (fluency + world knowledge)`

## Cursor kickoff prompt
> Implement general-corpus loaders (TinyStories, WikiText, OpenWebText subset) with WikiText
> weighted up, prepare `ctx_len=1024` shards, and pretrain the Small (~30–50M) model per
> `phases/08_phase8_general_pretraining.md`. Checkpoint frequently and report held-out
> perplexity plus a few factual-prompt completions to confirm world knowledge.
