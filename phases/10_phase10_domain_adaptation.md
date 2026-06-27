# Phase 10 — Domain Adaptation (Base on the LOGOS Corpus)

## Goal
Continue training into the **Base** (110–160M) model on the LOGOS corpus using curriculum
learning (general → philosophy), producing a fluent, on-topic domain model — the substrate
the VERITAS phases will shape.

## Prerequisites
Phase 8 (general pretrain), Phase 9 (corpus + holdout). Reproducible resume.

## Deliverables (files)
```
configs/base.yaml              # Base config (n_layer 12, n_head 12, d_model 768, ctx 2048)
training/curriculum.py          # general -> domain scheduling
scripts/train.py                # reused, with resume-from-checkpoint
experiments/experiment_00X/     # run record + held-out philosophy perplexity
```

## Spec
- Base tier ~110–160M, `ctx_len = 2048`, GQA optional, gradient checkpointing on.
- Curriculum: start from general weights (warm-start), blend toward philosophy.
- Validate on the **clean held-out philosophy set** from Phase 9 (never trained on).
- Optionally warm-start via the Medium (~75M) tier first (see `Technical_Spec.md` §4.5).

## Tasks
1. Write `configs/base.yaml`; enable gradient checkpointing.
2. Implement curriculum scheduling (ratio of general:domain shifting over training).
3. Warm-start from the general checkpoint; continue training. (Optionally grow Small → Medium
   → Base per the scaling strategy.)
4. Track held-out philosophy perplexity; watch for catastrophic forgetting of general fluency.

## Tests / checks
- Held-out philosophy perplexity hits target.
- General fluency retained (sanity prompts) — no catastrophic forgetting.
- Not overfit to corpus phrasing (samples generalize).

## Acceptance gate (from Expected_Outcome.md)
- Fluent, on-topic philosophy/religion/ethics completions; target perplexity on the held-out
  philosophy validation set; curriculum applied; no catastrophic forgetting.

## Definition of done / commit
`feat: domain-adapt Base model on LOGOS corpus with curriculum learning`

## Cursor kickoff prompt
> Domain-adapt the Base (~110–160M) model per `phases/10_phase10_domain_adaptation.md`: write
> `configs/base.yaml` (ctx 2048, grad checkpointing), implement general→philosophy curriculum
> scheduling, warm-start from the general checkpoint, and train while tracking held-out
> philosophy perplexity. Confirm no catastrophic forgetting of general fluency.
