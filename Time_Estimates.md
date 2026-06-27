# Project LOGOS — Time Estimates Per Phase

> Companion to `Detailed_Implementation_Plan.md` and `Expected_Outcome.md`.
> Estimates for how long each phase realistically takes.

## Assumptions (read first — your numbers shift if these change)

- **Team size:** one developer using Cursor as an implementation partner.
- **Effort basis:** "engineering days" = focused working days (~5–6 productive hours).
  Calendar time is usually longer.
- **Hardware:** Apple Silicon (MPS) or a single consumer GPU (e.g. one 8–24 GB card).
  This is the **main constraint** — training wall-clock dominates the late phases.
- **Two clocks matter:** *Build time* (writing/testing code) vs *Compute time* (waiting for
  training/eval to run, often unattended/overnight). They overlap — you can build the next
  phase while a model trains.
- Ranges are **Optimistic – Realistic – Cautious**. Use Realistic for planning.

---

## Quick Summary Table

| Phase | Build (dev-days) | Compute (wall-clock) | Realistic total |
|---|---|---|---|
| 1 Setup | 0.5 – 1 | — | ~1 day |
| 2 Tokenizer (BPE) | 2 – 4 | minutes–1 hr | ~3–4 days |
| 3 Dataset pipeline | 2 – 3 | minutes | ~2–3 days |
| 4 GPT architecture | 4 – 7 | minutes (overfit test) | ~1 week |
| 5 Training framework | 3 – 5 | hours (Tiny runs) | ~4–5 days |
| 6 Generation | 1 – 2 | minutes | ~2 days |
| 7 Tiny GPT (Shakespeare) | 1 – 2 | hours–1 day | ~2–3 days |
| 8 General pretrain | 2 – 4 | **days** | ~1 week+ |
| 9 Philosophy dataset builder | 4 – 8 | hours (cleaning) | ~1–2 weeks |
| 10 Domain adaptation | 2 – 4 | **days** | ~1 week+ |
| R Reasoning / CoT | 4 – 8 | 1–3 days | ~1.5–2 weeks |
| 11 Instruction dataset | 4 – 7 | hours | ~1–1.5 weeks |
| V Constitution + persona SFT | 3 – 6 | 1–2 days | ~1 week |
| P Anti-sycophancy DPO | 5 – 9 | 1–3 days | ~1.5–2 weeks |
| C Calibration / abstention | 3 – 5 | 1–2 days | ~1 week |
| 12 Evaluation suite | 4 – 7 | per-run (recurring) | ~1–1.5 weeks |
| 13 Inference API | 2 – 3 | — | ~2–3 days |
| 14 Chat app | 2 – 3 | — | ~2–3 days |
| 15 Experiment tracking | 1 – 2 | — | ~1–2 days |
| 16 Testing (90%+) | 3 – 5 | — | ~4–5 days |
| 17 Documentation | 2 – 4 | — | ~3–4 days |

**Rough end-to-end (solo, single GPU/MPS):**
- **Optimistic:** ~2.5 months
- **Realistic:** ~4–5 months
- **Cautious:** ~6–8 months

> The biggest variable is **data + compute**, not coding. Phases 8–10 and R/V/P/C are where
> calendar time balloons because of training runs, dataset curation, and the
> evaluate→tune→retrain loop.

---

## Phase-by-Phase Detail

### Foundation

**Phase 1 — Setup · ~1 day**
Mostly boilerplate. Cursor accelerates this heavily. Don't over-engineer tooling early.

**Phase 2 — Tokenizer (BPE from scratch) · ~3–4 days**
Build is straightforward; the time goes into correctness (lossless round-trip, special
tokens, unicode/whitespace edge cases) and training the vocab on your combined corpus.

**Phase 3 — Dataset pipeline · ~2–3 days**
Chunking, collation, and **loss masking** (the part that bites later if wrong). Worth
getting tests airtight here.

**Phase 4 — GPT architecture · ~1 week**
The densest *coding* phase. RoPE, RMSNorm, SwiGLU, attention, KV cache — each a tested
module. Budget extra for the **overfit-one-batch** debugging; a stubborn wiring bug here can
eat 1–2 extra days. Worth every hour.

**Phase 5 — Training framework · ~4–5 days**
Grad accumulation, mixed precision, scheduler, checkpoint/resume, reproducibility. The
reproducibility + resume correctness is the slow part.

**Phase 6 — Generation · ~2 days**
Sampling strategies + KV-cache parity with no-cache greedy. Quick once architecture is solid.

### Capability

**Phase 7 — Tiny GPT (Shakespeare) · ~2–3 days**
Mostly compute + a validation pass. First real "it works!" moment. Trains in hours on MPS/GPU.

**Phase 8 — General pretrain · ~1 week+**
Build is small; **compute dominates** (days of training for a 30–50M model on a single
device). Plan to build Phase 9 while this trains.

**Phase 9 — Philosophy dataset builder · ~1–2 weeks**
Usually **underestimated**. Sourcing, cleaning, dedup, metadata tagging (incl. `stance`/
`tradition`), and quality control of real books is slow, manual-ish work. This phase
directly determines how good the final model is — don't rush it.

**Phase 10 — Domain adaptation · ~1 week+**
Small code (curriculum + resume); **compute-bound** training days plus held-out validation.

### VERITAS (the character — expect iteration loops)

> These phases are not one-shot. Each involves a **build → train → evaluate → adjust →
> retrain** loop. The ranges already assume 1–2 iterations; more iterations = more time.

**Phase R — Reasoning / CoT · ~1.5–2 weeks**
Generating quality reasoning/fallacy data is the bulk; reasoning SFT adds 1–3 compute days.

**Phase 11 — Instruction dataset · ~1–1.5 weeks**
Authoring/curating persona-correct gold outputs + the style linter. Data quality work.

**Phase V — Constitution + persona SFT · ~1 week**
Constitution + system prompt drafting is fast; persona SFT + A/B evaluation is the time sink.

**Phase P — Anti-sycophancy + anti-fence-sitting DPO · ~1.5–2 weeks**
The hardest VERITAS phase. From-scratch DPO, preference-pair generation/labeling (chosen =
blunt verdict; rejected = flattery *and* both-sides mush), and **β/LR tuning** (mode-collapse
risk) make this iteration-heavy. Budget generously.

**Phase C — Calibration / earned credence · ~1 week**
ECE harness + contested-but-decidable set + calibration tuning. Moderate, but needs care to
get *honest* confidence — including lopsided credences — without sliding back into fake 50/50
neutrality.

### Verification

**Phase 12 — Evaluation suite · ~1–1.5 weeks (then recurring)**
Build the full truthfulness harness once; after that it **runs on every experiment**, so its
cost recurs throughout VERITAS phases. Building it *earlier* (alongside Phase R) pays off.

**Phase 13 — Inference API · ~2–3 days** · FastAPI endpoints + persona injection + KV cache.

**Phase 14 — Chat app · ~2–3 days** · Streamlit UI with reasoning/confidence UX.

**Phase 15 — Experiment tracking · ~1–2 days** · Auto-record runs/lineage (build early if possible).

**Phase 16 — Testing (90%+) · ~4–5 days** · Heaviest on DPO/calibration/masking tests.

**Phase 17 — Documentation · ~3–4 days** · Guides + `docs/veritas.md` incl. limitations.

---

## Scheduling Advice (cut calendar time without cutting corners)

1. **Overlap build and compute.** Always be coding the next phase while a model trains.
   This is the single biggest schedule saver.
2. **Build the eval harness (Phase 12) early** — around Phase R. You need it to judge every
   VERITAS iteration; building it last wastes earlier runs.
3. **Build experiment tracking (Phase 15) early** too, for the same reason.
4. **Front-load data work (Phases 9 & 11).** Data is the long pole and gates model quality.
5. **Expect 1–3 retrain loops** in P/V/C. Don't schedule them as one-shot.
6. **Renting cloud GPUs** for Phases 8/10/P can compress weeks of compute into days — the
   main lever if the timeline must shrink.

> Bottom line: with one developer on a single GPU/MPS, plan for **~4–5 months realistic**.
> Coding is rarely the bottleneck — **data curation and the train/evaluate/retrain loop are.**
