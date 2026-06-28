# Small general pretraining run — what happened

This document explains the **Phase 8** end-to-end training run: the Small (~30M) GPT pretrained
on a general corpus (TinyStories + WikiText-103 + an OpenWebText subset). It is written for
someone who wants to understand the pipeline, the numbers, and the artifacts — including an
**honest assessment of what passed and what did not** — without reading all the code.

**Canonical run:** `experiments/experiment_011/` (full 2500-step run)
**Bootstrap run:** `experiments/experiment_010/` (first attempt, early-stopped at step 200)
**Best weights:** `experiments/experiment_011/general_pretrain_small_best.pt`
**Config used:** `configs/small.yaml` (copied into each experiment folder as `config.yaml`)

---

## What we were trying to do

Pretrain a Small GPT **from scratch** on a mixed general corpus to get **fluency *and* basic
world knowledge** — the factual ground the model must later be "truthful about" (per
`phases/08_phase8_general_pretraining.md`):

- validation perplexity reaches a reasonable target on a general held-out set
- generated completions are coherent general-purpose English
- the **expository/factual slice** (WikiText) is well represented — not fluent-but-empty
- the model is not just copying long verbatim spans from training data

This is **not** a chat or Q&A model. It learns to **continue text** in the style of the corpus.

---

## End-to-end pipeline

```mermaid
flowchart LR
  A[Fetch sources via HF parquet] --> B[Clean + weight mix]
  B --> C[Train BPE tokenizer vocab 8192]
  C --> D[Pack into 1024-token shards]
  D --> E[Initialize Small GPT ~29M params]
  E --> F[Train AdamW + cosine LR on MPS]
  F --> G[Eval every 100 steps]
  G --> H[Checkpoint every 250 steps]
  H --> I[Generate factual samples + metrics]
```

**Commands that produced the runs:**

```bash
# First attempt (early-stopped at step 200)
python scripts/train.py --config configs/small.yaml

# Full run (resumed from the step-200 checkpoint, early stop disabled)
python scripts/train.py --config configs/small.yaml \
  --resume experiments/experiment_010/general_pretrain_small_best.pt
```

---

## 1. Data

| Item | Value |
|------|--------|
| Sources | TinyStories, WikiText-103-raw, OpenWebText (subset) |
| Ingest | **Direct HuggingFace parquet downloads** (`datasets/sources/general.py`) |
| Local cache | `data/general/{wikitext,tinystories,openwebtext}/parquet/...` |
| Mix weights | WikiText **×5**, TinyStories ×1, OpenWebText ×1 |
| Raw caps | WikiText 100k docs · TinyStories 50k · OpenWebText 10k (+ val) |
| Weighted train docs | **560,000** (500k WikiText after ×5 + 50k + 10k) |
| Val docs | 8,352 (held-out from all three sources) |
| Train sequences | **136,781** packed blocks |
| Val sequences | **3,673** packed blocks |
| Context length | **1024** tokens per block |

**WikiText is deliberately upweighted ×5** so the factual/expository slice dominates — the
phase spec warns that fluency without facts produces "a model that is confidently empty."

> **Ingest note:** the original loader streamed the HuggingFace *rows API* and stalled for
> ~30 min on HTTP 429 rate limits (and a 502 on OpenWebText) without ever reaching training.
> It was rewritten to download **parquet shards** directly (same pattern as WikiText). Full
> data prep then dropped to **~14 s of fetch + ~3.5 min of packing**.

---

## 2. Tokenizer

| Item | Value |
|------|--------|
| Type | Byte-level BPE (built in-repo, Phase 2) |
| Vocab size | 8192 (+ 12 special tokens) |
| Trained on | Bounded mixed sample (~8M chars) from the weighted corpus |
| Saved to | `data/general/tokenizer/` (shared) and each run's `tokenizer/` |

The tokenizer is trained once and cached for reruns. **Do not change `vocab_size`** after it is
fixed without re-tokenizing the corpus.

---

## 3. Model (Small tier)

Settings from `configs/small.yaml`:

| Hyperparameter | Value |
|----------------|--------|
| Layers | 8 |
| Heads | 8 |
| `d_model` | 512 |
| Context | 1024 |
| Architecture | Decoder-only, RoPE, RMSNorm, SwiGLU, weight tying |
| Dropout | 0.1 |
| **Parameter count** | **29.36M** (in the 30–50M "Small" band) |

---

## 4. Training setup

| Setting | Value |
|---------|--------|
| Optimizer | AdamW (`lr=2.5e-4`, `weight_decay=0.1`) |
| Scheduler | Cosine decay with 200-step warmup |
| Batch size | 2 (**effective 32** with `grad_accum_steps=16`) |
| Max steps | 2500 |
| Precision | bfloat16 on Apple MPS |
| Eval interval | Every 100 steps |
| Checkpoint interval | Every 250 steps |
| Seed | 42 (reproducible) |
| Throughput | **~8.2 s/step** on MPS (~7h46m for the resumed 2300 steps) |

**Loss:** masked cross-entropy over all non-padding next-token positions.
**Perplexity:** `exp(eval_loss)` — lower is better.

> **Important:** effective batch 32 × 2500 steps = **80,000 sequences seen**, but the dataset has
> **136,781** sequences. The model trained for **< 1 epoch** (~0.58) — roughly **82M tokens**,
> about **14% of compute-optimal** (~580M for a 29M model). This matters for interpreting the
> results below.

---

## 5. What happened during training (loss curve)

The first run (`experiment_010`) used `early_stop: true` with target perplexity **75.0**. It
crossed the target at **step 200** (val PPL 53.85) and stopped — far too early, before learning
any facts. We set `early_stop: false`, lowered the target to 40.0, and **resumed** to run the
full schedule (`experiment_011`).

Validation milestones from `experiment_011/loss.csv`:

| Step | Val perplexity |
|------|----------------|
| 300 | 37.72 |
| 500 | 26.51 |
| 1000 | 18.47 |
| 1500 | ~15.8 |
| 1900 | 15.36 |
| 2300 | 15.16 |
| 2400 | 15.14 |
| **2500** | **15.13** |

Perplexity fell fast to ~18.5 by step 1000, then **flattened at ~15** from step 1900 onward.
The flattening tracks the **cosine LR decaying toward zero** (LR ≈ 2.5e-5 at the end) — the model
was still under one epoch and the train/val gap stayed small (train PPL ~13.5 vs val 15.1), so it
was **not overfit and not at a hard ceiling**; the schedule simply ran out of learning rate.

---

## 6. Final results

From `experiments/experiment_011/metrics.json`:

| Metric | Value |
|--------|--------|
| Steps | 2500 |
| Best step | 2500 |
| Final / best val loss | 2.717 |
| **Final / best val perplexity** | **15.13** |
| Target | 40.0 — **passed** |
| Parameter count | 29,364,736 |
| Memorization spot-check | **No** long verbatim spans in sample outputs |

### Gate assessment (honest)

| Phase 8 criterion | Status |
|-------------------|--------|
| Small ~30–50M, ctx 1024 | ✅ 29.36M |
| Loaders + weighted WikiText | ✅ |
| Packed 1024-token shards | ✅ |
| Held-out perplexity target | ✅ 15.13 ≪ 40 |
| Coherent general completions | ⚠️ Partial — fluent, often nonsensical |
| Basic world knowledge on factual prompts | ❌ Not met |
| Expository slice well represented (at inference) | ❌ Not met |

**Bottom line: the perplexity gate passed; the world-knowledge gate did not.** This is the exact
"fluent but factually empty" red flag the phase spec warns about.

### Sample quality (honest assessment)

Factual prompts from `experiments/experiment_011/factual_samples.txt`:

- *"The capital of France is the first of the first two species of the first two species…"*
- *"Water boils at the end of the season. The game was released on September 1, 2010…"*
- *"Einstein is best known for the first time in the United States…"*

The model produces **WikiText-flavored prose** but **does not recall facts**. Three compounding
causes, in order of impact:

1. **Schedule + undertraining** — < 1 epoch and LR decayed to ~0 mid-training (see §4–5).
2. **Decoding artifact** — greedy (`temperature=0.0`) with no repetition penalty loops badly on a
   small LM; this inflates how broken it looks.
3. **Data/eval mismatch** — facts like "water boils at 100°C" may simply **not appear** in
   WikiText-103, and TinyStories adds fluency filler with zero factual content.

So the failure is **not** primarily "29M is too small" — it is mostly **how it was trained and
measured**. See `docs/phase8_next_steps.md` for the plan going forward.

---

## 7. Artifacts in `experiments/experiment_011/`

| File | Purpose |
|------|---------|
| `config.yaml` | Exact config snapshot for this run |
| `loss.csv` | Step-by-step train loss; eval loss/perplexity every 100 steps |
| `metrics.json` | Final numbers + memorization check |
| `samples.txt` / `factual_samples.txt` | Generated text at temperatures 0.0 / 0.7 |
| `tokenizer/` | BPE vocab for this run |
| `general_pretrain_small_step_{250…2500}.pt` | Periodic checkpoints (every 250 steps) |
| **`general_pretrain_small_best.pt`** | **Best validation weights** |
| `general_pretrain_small.pt` | Final export (same weights after restoring best) |

---

## 8. How to test the model

**CLI:**

```bash
python scripts/generate.py \
  --checkpoint experiments/experiment_011/general_pretrain_small_best.pt \
  --tokenizer experiments/experiment_011/tokenizer \
  --prompt "The history of the Roman Empire" \
  --temperature 0.7
```

Use **open-ended prose prompts**, not trivia questions. Plain factual queries ("What is the
capital of France?") will not answer reliably — this checkpoint is a **fluency base**, not a
knowledge base.

---

## 9. How to retrain

```bash
make train-small
# or with explicit config:
python scripts/train.py --config configs/small.yaml
```

Each run creates a new `experiments/experiment_NNN/` folder. Data prep and tokenizer are cached
under `data/general/`, so reruns start at packing/training. To rebuild shards from scratch:

```bash
make prepare-general            # or: python scripts/prepare_general.py --force
```

---

## 10. What comes next

See **`docs/phase8_next_steps.md`** — the single plan going forward: implement fair eval,
re-score this checkpoint, close Phase 8, start Phase 9. No corrective Small retrain unless
re-scoring shows a surprise hit-rate worth pushing.
