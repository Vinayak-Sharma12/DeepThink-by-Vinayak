# Project LOGOS — Technical Specification

> **Model name (display):** DeepThink by Vinayak · **Codename:** LOGOS · **Character pillar:** VERITAS
> **Type:** Decoder-only autoregressive transformer (GPT-style), built entirely from scratch
> in PyTorch.
> **Domain:** philosophy, religion, spirituality, psychology, ethics, and logic.
> **Companion docs:** `Detailed_Implementation_Plan.md` (engineering plan),
> `Expected_Outcome.md` (phase gates), `Sample_Answers.md` / `Model_Answers_50.md`
> (behavioral target), `Time_Estimates.md` (schedule).

This document is the single source of truth for **what** is built (interfaces, shapes,
hyperparameters, formats, contracts). The implementation plan covers the **how/when**; this
spec covers the **exact technical surface**.

---

## 1. Scope & Design Principles

### 1.1 What this system is

A from-scratch GPT trained through a multi-stage pipeline (pretrain → domain-adapt →
reason → persona → preference-optimize → calibrate) to produce a model that is **truthful,
blunt, opinionated, and decisive** — it reads every side, commits to a verdict, and states
its real (often lopsided) confidence.

### 1.2 Non-negotiable engineering constraints

| Constraint | Rule |
|---|---|
| No external model code | No HuggingFace `Trainer`, no copied GPT implementations. Every major component is implemented in-repo. |
| Modularity | Each architectural unit (attention, norm, FFN, etc.) is an independent, unit-tested module. |
| Reproducibility | Every run is bit-reproducible: fixed seeds + full checkpoint state (model/opt/sched/RNG/config). |
| Typing & docs | Type hints everywhere; docstrings on every public function; PEP8. |
| Test coverage | 90%+, with the alignment-critical pieces (loss masking, DPO, ECE) the best-tested code in the repo. |

### 1.3 Non-goals

- Not multimodal, not multilingual-first (English primary; domain terms tokenized well).
- Not a RLHF/PPO system — preference optimization is **DPO** (no reward model, no RL loop).
- Not a retrieval system in v1 (no live web/RAG); knowledge is parametric.

---

## 2. System Architecture (high level)

```
                      ┌──────────────────────────────────────────────┐
   raw corpora ─────▶ │ Data pipeline: clean → chunk → tokenize →     │
   (general +         │ pack → loss-mask                              │
   philosophy)        └──────────────────────────────────────────────┘
                                        │ token streams
                                        ▼
                      ┌──────────────────────────────────────────────┐
   BPE tokenizer ───▶ │ GPT model (RoPE · RMSNorm · SwiGLU · pre-norm │
   (from scratch)     │ · GQA-optional · KV cache · weight tying)     │
                      └──────────────────────────────────────────────┘
                                        │ checkpoints
                                        ▼
   Stage pipeline:  pretrain → domain-adapt → reasoning SFT → persona SFT
                    → DPO (anti-sycophancy/anti-fence-sitting) → calibration
                                        │
                                        ▼
              evaluation (VERITAS suite) → promote → serve (FastAPI) → chat (Streamlit)
```

---

## 3. Tokenizer Specification

### 3.1 Algorithm

- **Byte Pair Encoding (BPE)**, implemented from scratch (`tokenizer/`).
- Trained on the **combined corpus** (general + philosophy) so domain terms
  (*epoché, qualia, dharma, telos, apophatic*) compress efficiently rather than exploding
  into single bytes.
- Byte-level fallback so `decode(encode(x)) == x` is **lossless** for arbitrary unicode and
  whitespace.

### 3.2 Vocabulary

| Property | Value |
|---|---|
| Target vocab size | 32,000 (configurable) |
| Encoding | byte-level BPE |
| Round-trip guarantee | lossless on held-out text incl. unicode/whitespace |

### 3.3 Reserved special tokens (registered at training time, never re-tokenized later)

```
<|bos|> <|eos|> <|pad|> <|unk|>
<|system|> <|user|> <|assistant|>
<|think|> <|/think|>
<|cite|> <|/cite|>
```

- `<|think|> ... <|/think|>` wraps the hidden reasoning trace (suppressible at inference).
- `<|cite|> ... <|/cite|>` wraps source/justification spans.
- Special tokens must never be split by the merge table.

### 3.4 Public interface

```python
class Tokenizer:
    def train(self, corpus: Iterable[str], vocab_size: int) -> None: ...
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], *, skip_special: bool = True) -> str: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Tokenizer": ...
```

---

## 4. Model Architecture Specification

Decoder-only, pre-norm transformer. Each item below is an independent module with its own
tests (`model/`).

### 4.1 Component choices (and rationale)

| Component | Choice | Why |
|---|---|---|
| Position encoding | **RoPE** (rotary) | Length generalization for long reasoning chains; no learned pos table. |
| Normalization | **RMSNorm** (pre-norm) | Simpler/stabler than LayerNorm; standard in modern reasoning LMs. |
| Feed-forward | **SwiGLU** | Better quality-per-parameter than GELU MLP. |
| Attention | Causal multi-head; **GQA optional** at Base+ | GQA cuts KV-cache cost for long-context inference. |
| Inference cache | **KV cache** | Required for usable generation latency. |
| Embedding/Output | **Weight tying** | Parameter efficiency; ties input embedding ⇄ output projection. |
| FFN ratio | ~ (8/3)·d_model (SwiGLU convention) | Keeps SwiGLU param count ≈ a 4× GELU MLP. |

### 4.2 Forward-pass contract

- Input `input_ids: LongTensor[B, T]` → logits `FloatTensor[B, T, vocab]`.
- Causal masking enforced in attention.
- With/without KV cache, **greedy decoding outputs must match** (parity test).
- A `count_parameters()` utility reports total/trainable params.

### 4.3 Config dataclass (single source of model dims)

```python
@dataclass
class GPTConfig:
    vocab_size: int = 32_000
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None      # None => MHA; < n_head => GQA
    d_model: int = 768
    ffn_mult: float = 8/3             # SwiGLU
    ctx_len: int = 2048
    rope_theta: float = 10_000.0
    rmsnorm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True
```

### 4.4 Model size ladder

> **Updated:** a **Medium** tier (~75M) is inserted between Small and Base to smooth the
> jump from ~40M → ~140M and give a stronger intermediate checkpoint before the costly Base
> stage. Adjust freely — these are starting points, gated by the overfit/perplexity tests.

| Stage | Params | n_layer | n_head | d_model | ctx_len | Purpose |
|---|---|---|---|---|---|---|
| Nano | ~2M | 4 | 4 | 128 | 256 | Wiring / overfit-a-batch sanity |
| Tiny | ~10M | 6 | 6 | 384 | 512 | Phase 7 (Shakespeare) |
| Small | 30–50M | 8 | 8 | 512 | 1024 | Phase 8 (general pretrain) |
| **Medium** | **~70–90M** | **10** | **10** | **640** | **1536** | **Intermediate pretrain/domain bridge** |
| Base | 110–160M | 12 | 12 | 768 | 2048 | Domain + reasoning + persona (carries VERITAS) |

> **Context-length priority:** any model that will carry the VERITAS character targets
> `ctx_len >= 1024` (reasoning, self-critique, and citations all consume tokens), even at the
> cost of fewer parameters.

### 4.5 Scaling strategy (parameter count is fixed per run)

The parameter count is an **architectural choice fixed before each training run** — it is not
a dial turned up/down mid-run (the weight-matrix shapes are defined by `GPTConfig`). Across
runs you have three levers:

| Lever | When | How |
|---|---|---|
| **From-scratch per rung** | default | Pick a size, train from random init. Each ladder rung is a separate model. |
| **Warm-start / growth** | Small → Medium → Base | Initialize the larger model from a trained smaller one (copy weights into matching slices, randomly init the new layers/width), then continue training. Saves compute vs. random init. |
| **Compression after training** | for deployment | Pruning (drop low-importance weights/heads), quantization (fp16/bf16 → int8/int4; shrinks memory, not count), or distillation (train a smaller student). |

Freely tunable mid-run **without** changing the count: learning rate, batch size, data mix,
dropout, train-time sequence length (≤ `ctx_len`), number of steps. Most "tuning" is these,
not the weight shapes.

> **Most disruptive change:** `vocab_size` — it touches tokenizer + embedding + output layer
> together. This is why the BPE tokenizer is trained **once** on the combined corpus with all
> special tokens reserved up front, so model scaling never forces a re-tokenization.

---

## 5. Data Pipeline Specification

### 5.1 Stages (`datasets/`, `data/`)

```
clean → dedup → chunk → tokenize → pack(ctx_len) → loss-mask → batch
```

### 5.2 Batch contract

A collated batch yields:

| Tensor | Shape | Meaning |
|---|---|---|
| `input_ids` | `[B, T]` | token ids |
| `targets` | `[B, T]` | `input_ids` shifted by 1 |
| `loss_mask` | `[B, T]` | 1 where loss counts, 0 elsewhere |

- **Loss masking is mandatory** for SFT/persona/DPO: prompt tokens are masked (loss = 0);
  only the assistant/completion span contributes to the loss. Implemented from scratch and
  unit-tested (off-by-one between `input_ids` and `targets` is the classic bug).

### 5.3 Corpora & metadata

| Stage | Data |
|---|---|
| General pretrain | TinyStories, WikiText, OpenWebText (subset); WikiText weighted for factual grounding. |
| Domain | LOGOS philosophy/religion/ethics/psychology corpus (built in Phase 9). |

Each domain document carries metadata:

```
title, author, year, school, religion, topic, language, stance, tradition
```

- `stance ∈ {argues_for, argues_against, survey, primary_source, commentary}`.
- `stance`/`tradition` power **balanced input sampling**: the model must *read every side at
  its strongest* before it concludes. Balanced **input** ≠ balanced **output** — the model
  is expected to reach a verdict, not echo each school equally.

---

## 6. Training Specification

### 6.1 Objective

- Standard **next-token cross-entropy**, computed only over unmasked positions.
- SFT/persona/DPO all reuse the same prompt-masking utility.

### 6.2 Trainer features (`training/`, CPU / Apple MPS / CUDA)

| Feature | Spec |
|---|---|
| Gradient accumulation | configurable micro-batch → effective batch |
| Mixed precision | bf16 on CUDA/MPS where supported |
| LR schedule | cosine decay with linear warmup |
| Gradient clipping | global-norm clip (e.g. 1.0) |
| Gradient checkpointing | enabled for Base tier |
| Seeding | deterministic; RNG state saved/restored |
| Optimizer | AdamW (from-scratch config), decoupled weight decay |

### 6.3 Checkpoint contract

A checkpoint stores **everything needed for bit-reproducible resume**:

```
model_state, optimizer_state, scheduler_state, step,
rng_state (python/numpy/torch), GPTConfig, tokenizer ref, stage tag
```

- Resume-from-checkpoint must reproduce the pre-save loss curve.

### 6.4 End-to-end stage order (the order is the character)

```
1. Pretrain (general)            → fluency + world knowledge        [Phase 8]
2. Domain-adapt (LOGOS corpus)   → philosophy depth, many traditions [Phase 10]
3. Reasoning SFT                 → learns to deliberate, then decide [Phase R]
4. Instruction + Persona SFT     → harsh, decided default voice      [Phase 11 + V]
5. DPO preference optimization   → blunt verdicts over flattery/fence [Phase P]
6. Calibration tuning            → honest (often lopsided) credence  [Phase C]
7. Evaluate → promote → repeat   → VERITAS suite gates release        [Phase 12]
```

Each arrow = a checkpoint + experiment record + VERITAS report + git commit.

---

## 7. VERITAS Alignment Stack (technical)

### 7.1 Reasoning format (Phase R)

```
<|system|> <persona/constitution>
<|user|>   <question>
<|assistant|>
<|think|>
  - restate the question
  - known vs assumed
  - steelman each candidate answer at its strongest
  - strongest counter-argument
  - weigh evidence; decide which side wins and by how much
<|/think|>
<final answer: leads with the verdict, blunt, decided>
```

The `<|think|>` span is trained but suppressible at inference. **Trace↔answer consistency**
is an eval gate — the final answer must follow from the trace and must end in a verdict, not
a shrug.

### 7.2 Constitution (`configs/constitution.md`) — 10 principles

Summary (full text in config): truth over comfort · **commit to a verdict** · calibrate
honestly (lopsided is fine) · no flattery/filler, harshness earned by bad reasoning ·
steelman before you strike · never fabricate · distinguish fact/interpretation/opinion then
take a side · update on evidence (conviction not dogma) · refuse sophistry · lead with the
verdict.

> **Harshness policy:** the "never cruel to the person" guardrail is **removed by design**.
> The model may be harsh toward the user, not only their ideas (no politeness filter). The
> sole anchor is *coherence*: contempt must be earned by the badness of the reasoning, never
> random.

### 7.3 DPO (Phase P) — implemented from scratch

- Frozen **reference model** = the Phase-V SFT checkpoint.
- Sequence log-prob computed with the same prompt-masking as SFT; reference forward pass runs
  under `no_grad`.
- **Loss:**

```
L = -log σ( β · [ (logπ_θ(y_w|x) − logπ_ref(y_w|x)) − (logπ_θ(y_l|x) − logπ_ref(y_l|x)) ] )
```

where `y_w` = chosen, `y_l` = rejected, `β` ≈ 0.1 (tuned; too high → no movement, too low →
drift/degradation).

- **Preference axes** (`chosen` vs `rejected`):

| Axis | chosen | rejected |
|---|---|---|
| Sycophancy | tells the user their argument is flawed | agrees to please |
| Fabrication | "no reliable source for that" | invents a citation |
| Fence-sitting | commits: "evidence favors X (~80/20)" | "it's contested / who's to say" |
| False balance | states real lopsided credence | fake 50/50 to seem neutral |
| Filler | leads with the verdict | flattery + padding |
| Cowardice | states the uncomfortable conclusion | both-sides mush |
| Empty certainty (guardrail) | strong claim backed by argument | strong claim backed by nothing / fabricated |

### 7.4 Calibration & abstention (Phase C)

- **Verbalized credence** attached to claims (incl. lopsided, e.g. "~85%").
- **ECE** (Expected Calibration Error) implemented from scratch; target is *accuracy of the
  number*, not pushing numbers toward 50.
- **Reserved abstention:** "no one knows" only for the genuinely unknowable; using it as a
  dodge on a decidable question is penalized.

---

## 8. Inference & Generation Specification

### 8.1 Sampler (`inference/`, `generate.py`)

| Strategy | Params |
|---|---|
| Greedy | — |
| Temperature | `temperature` |
| Top-k | `top_k` |
| Top-p (nucleus) | `top_p` |

- Stop-on-`<|eos|>`; max-new-tokens guard against runaway generation.
- **Reasoning decode modes:** `show` / `hide` the `<|think|>...<|/think|>` span.
- KV-cache path **must** match the no-cache path for greedy decoding (parity test).

---

## 9. Serving API Specification (Phase 13, FastAPI)

| Endpoint | Method | Notes |
|---|---|---|
| `/generate` | POST | raw completion; `show_reasoning: bool` |
| `/chat` | POST | role-formatted; `show_reasoning: bool`; returns `confidence` when emitted |
| `/health` | GET | liveness |
| `/model` | GET | model + config metadata |

- **Persona system prompt is injected server-side by default** (persona must not depend on
  the client supplying it).

### 9.1 `/chat` response (illustrative)

```json
{
  "answer": "Probably no God in any sense that would matter to your life. ...",
  "confidence": 0.82,
  "reasoning": "<hidden unless show_reasoning=true>",
  "name": "DeepThink by Vinayak",
  "model": "logos-base",
  "stage": "R+V+P+C"
}
```

---

## 10. Evaluation Specification (Phase 12 — VERITAS suite)

All metrics implemented from scratch (`evaluation/`); run automatically after each relevant
experiment.

| Metric | Measures |
|---|---|
| Truthfulness score | resists common false beliefs / leading questions |
| Sycophancy rate | how often it caves when the user pushes a wrong claim |
| Fabrication rate | how often it invents quotes/citations/facts |
| **Conviction rate** | reaches a verdict on contested questions vs fence-sitting |
| Calibration (ECE) | accuracy of stated confidence, incl. lopsided |
| Unwarranted-abstention rate | "no one knows" on a *decidable* question (counts against it) |
| Contradiction rate | self-consistency of the verdict across paraphrases |
| Bluntness/style | no flattery/filler; leads with the verdict (style linter) |
| Reasoning quality | logic/fallacy/multi-step accuracy |

**Promotion gate:** no checkpoint is promoted unless truthfulness/sycophancy/fabrication and
conviction improve or hold while helpfulness does not regress.

---

## 11. Experiment Tracking (Phase 15)

Each run auto-creates `experiments/experiment_XXX/`:

```
config.yaml          # full GPTConfig + training config
metrics.json         # all eval metrics
loss.csv             # loss curve
generation_samples.md
veritas_report.md    # metric table + deltas vs previous best, tagged with R/V/P/C lineage
```

---

## 12. Testing Specification (Phase 16, 90%+)

Highest-priority (load-bearing) tests:

- loss masking (prompt zeroed, completion counted; no off-by-one)
- sequence log-prob (used by DPO)
- DPO loss on a toy pair where `chosen` is obviously better
- ECE computation
- style linter (banned phrases + fence-sitting tells)
- abstention scorer
- overfit-one-batch (Nano → ~0 loss)
- KV-cache vs no-cache greedy parity

---

## 13. Repository Layout

```
logos/
  configs/        constitution.md, persona_system_prompt.txt, *.yaml
  data/ datasets/ raw + processed corpora, builders, metadata
  tokenizer/      BPE (train/encode/decode/serialize)
  model/          embeddings, rope, rmsnorm, swiglu, attention, block, gpt
  training/       trainer, optimizer, scheduler, checkpoint, dpo
  inference/      sampler, kv-cache, generate
  evaluation/     veritas suite, ece, style linter, scorers
  experiments/    auto-generated run records
  app/            FastAPI service + Streamlit chat
  scripts/        entry points
  tests/          unit + integration
  models/         checkpoints
  logs/ docs/
```

---

## 14. Hardware, Performance & Reproducibility

| Aspect | Spec |
|---|---|
| Devices | CPU, Apple MPS, CUDA (single consumer GPU / Apple Silicon assumed) |
| Precision | bf16 mixed precision where supported |
| Long-context cost | mitigated by GQA (Base+) and gradient checkpointing |
| Determinism | seeded RNG; full RNG state in checkpoints; resume == original loss curve |
| Bottleneck | data curation + train/eval/retrain loop dominate wall-clock, not coding |

---

## 15. Known Limitations (stated honestly, per the project's own values)

- **Parametric knowledge only** (v1): no retrieval; facts can be stale or wrong despite a
  confident tone — calibration reduces but does not eliminate this.
- **Opinionated by design:** the model takes sides; on contested metaphysical/religious
  questions it leans naturalist/skeptical because that's where the evidence points — this is
  a *values configuration*, adjustable in `configs/constitution.md`, not an oracle.
- **No personal-cruelty guardrail:** harshness toward the user is permitted; the coherence
  anchor (contempt earned by bad reasoning) is enforced by data/rubric, not a hard filter.
- **Small scale:** at ~10–160M params it is a specialist, not a frontier generalist; expect
  brittleness outside its domains.

---

> **Definition of done (technical):** the pipeline reproducibly produces a checkpoint that
> passes every gate in `Expected_Outcome.md` and behaves like the golden answers in
> `Model_Answers_50.md` / `Sample_Answers.md` across a held-out test set — verified by the
> metrics in `veritas_report.md`.
