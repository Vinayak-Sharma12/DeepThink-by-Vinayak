# Phase 8 — Status and Next Steps

Single reference after general-pretraining runs (`experiment_010`, `experiment_011`). For run
history, numbers, and artifacts see `docs/general_pretraining_small.md`.

**Phase 8 is closed on paper.** Pipeline, fluency checkpoint, and fair evaluation are done.
**Next work: Phase 9** (philosophy corpus).

---

## 1. Where we stand

| Done | Not done |
|------|----------|
| General-pretrain pipeline (parquet loaders, shards, train, checkpoint) | Phase 9 philosophy corpus |
| Fluency — val PPL **15.13** (`experiment_011`) | Medium/Base pretrain (Phase 10) |
| Fair factual evaluation (W4 + W5) | |
| Knowledge gate measured honestly — **0/25 (0%)** on fair decode | |
| Checkpoint `experiments/experiment_011/general_pretrain_small_best.pt` | |

The first runs failed the knowledge gate on **unfair** probes (greedy decode, repetition loops).
Fair re-scoring confirms: fluent but no factual retrieval at Small scale.

---

## 2. Fair eval — completed

Fix three measurement bugs:

| Bug | Fix | Status |
|-----|-----|--------|
| Greedy decode loops | **Repetition penalty** in sampler (W4) | ✅ |
| Unfair trivia probes | **In-corpus** probes only (W5) | ✅ |
| PPL-only gate | Report **`factual_probe_hit_rate`** alongside PPL | ✅ |

### 2.1 W4 — Repetition penalty

| File | Change |
|------|--------|
| `inference/sampler.py` | `repetition_penalty` on `SamplerConfig`, `apply_repetition_penalty`, `sample_token(..., prev_token_ids=...)` |
| `inference/generate.py` | Pass token history into `sample_token` |
| `tests/test_generation.py` | `test_repetition_penalty_is_noop_at_one`, `test_repetition_penalty_downweights_seen_token`; KV-cache parity still passes at `penalty=1.0` |

### 2.2 W5 — Factual probes

| File | Purpose |
|------|---------|
| `evaluation/factual_probes.py` | `load_probes`, `verify_in_corpus`, `completion_hits`, `score_factual_probes`, `write_probe_results` |
| `evaluation/data/factual_probes.json` | 25 probes; answers verified in WikiText/OWT |
| `scripts/score_factual_probes.py` | Re-score any checkpoint without retraining |
| `tests/test_factual_probes.py` | Unit tests |

**Probe schema:**

```json
{"prompt": "The capital of France is", "answers": ["Paris"], "source": "wikitext"}
```

**Fair decoder defaults:** `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.3`.

**Hit:** any `answers` substring appears in the completion (case-insensitive).

### 2.3 Integration

| File | Change |
|------|--------|
| `scripts/train.py` | `build_sampler_config()` threads `top_p` / `repetition_penalty`; general runs write `factual_probes.json` and probe metrics into `metrics.json` |
| `configs/small.yaml` | `top_p`, `repetition_penalty`, `fair_eval_temperature`, `fair_eval_max_new_tokens`, `factual_probes_path` |

Future general train runs log probe hit-rate automatically. Re-score an existing checkpoint:

```bash
python scripts/score_factual_probes.py \
  --checkpoint experiments/experiment_011/general_pretrain_small_best.pt \
  --tokenizer experiments/experiment_011/tokenizer
```

---

## 3. Re-score result (`experiment_011`)

| Metric | Value |
|--------|-------|
| Val perplexity | **15.13** |
| Probes kept (in-corpus) | 25 / 25 |
| Factual probe hit-rate | **0 / 25 (0%)** |
| Fair decode | `temp=0.7`, `top_p=0.9`, `repetition_penalty=1.3`, `max_new_tokens=48` |
| Artifacts | `experiments/experiment_011/factual_probes.json` |

**Verdict:** fluency gate passed; knowledge gate failed at Small scale. Completions are fluent
but wrong (e.g. “The capital of France is a member of the Empire's commandors”). This is
expected — deep world knowledge belongs at **Base scale (Phase 10)**, not ~30M Small.

No corrective 5000-step Small retrain unless you explicitly want to chase knowledge at this
scale (re-score would need **≥ 40%** to justify it).

---

## 4. Extras beyond the original plan

These were not in the original checklist but landed as part of implementation:

| Extra | Why |
|-------|-----|
| `build_mix_config()` in `datasets/sources/general.py` | Shared by `train.py`, `prepare_general.py`, and `score_factual_probes.py` — avoids duplicating YAML→`CorpusMixConfig` parsing |
| `build_sampler_config()` in `scripts/train.py` | Single helper for sampling + fair-eval sampler construction from YAML |
| `completion_hits()` exported from `evaluation/factual_probes.py` | Separates hit logic from generation loop; unit-tested directly |
| `fair_eval_max_new_tokens: 48` in `configs/small.yaml` | Shorter cap for probe scoring than demo sampling (`max_new_tokens: 80`) |
| `factual_probe_hits` / `factual_probe_total` in `metrics.json` | Raw counts alongside `factual_probe_hit_rate` |
| `--verify-corpus` / `--no-verify-corpus` on score script | CLI control over in-corpus probe filtering (default: verify) |

All **88 unit tests** pass.

---

## 5. Phase 8 closure

| Hit-rate | Action | **Our result** |
|----------|--------|----------------|
| **≥ 50%** | Knowledge gate passed | |
| **20–50%** | Document "fluency + partial knowledge" | |
| **< 20%** | Document "fluency base; knowledge deferred to Base" | **← 0%** |

Keep `experiment_011` as the fluency checkpoint. Suggested commit (when ready):

```
feat: general pretraining pipeline + Small fluency checkpoint (Phase 8)
```

---

## 6. Next — Phase 9

Start `phases/09_phase9_philosophy_dataset.md`:

- `datasets/corpus/ingest.py`, `clean.py`, `dedup.py`, `metadata.py`
- JSONL with `stance`, `tradition`, `school`, …
- `DATA_SOURCES.md` + held-out val set under `data/logos/holdout/`

Phase 9 is what makes LOGOS substantive. It can run in parallel with any future pretrain.

---

## 7. Explicitly out of scope (for now)

- Full corrective Small retrain (5000 steps, 45M, new mix)
- Chasing PPL below 15
- Medium/Base pretrain (comes after Phase 9 corpus exists)

Apply expository-heavy mix lessons (`WikiText up`, `TinyStories down`) at **Medium/Base
pretrain**, not another Small run.
