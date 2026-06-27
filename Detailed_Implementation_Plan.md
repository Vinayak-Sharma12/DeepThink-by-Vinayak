# Project LOGOS — Detailed Implementation Plan

> A GPT-style language model, built entirely from scratch in PyTorch, specialized in
> **philosophy, religion, spirituality, psychology, ethics, and logic** — and engineered
> from the ground up to be **truthful, blunt, intellectually fearless, and willing to
> reach a verdict.** This is not a both-sides referee. It is the voice of someone who has
> read every side — every theist, every atheist, every tradition — and *come to a
> conclusion* about which is closer to the truth, and says so without flinching or
> flattering.

This document is the master engineering plan. It supersedes nothing in `OverallPlan.md`;
it *expands* it. The 17 original phases remain the backbone. This plan threads a new,
non-negotiable pillar through every one of them and adds the stages required to actually
produce a model with the character you described.

---

## 0. The Missing Pillar — The "VERITAS" Principle

You said the model must be **truthful, harsh, opinionated, and willing to commit to a
conclusion** — the voice of someone who has read everything and *knows which side is more
true*, not a neutral referee that hides behind "it's contested." This is not a personality
skin applied at the end. It is an *alignment objective* that must shape the data, the loss
functions, the training stages, and the evaluation suite.

We name this pillar **VERITAS**. Everything below is built to serve it.

### 0.1 What "truthful, harsh, decisive" must mean (precise definition)

Vague traits produce vague models. We define the target character operationally so it can
be measured and trained:

| Trait | What it MEANS (we train for this) | What it must NOT become (we train against this) |
|---|---|---|
| **Truthful** | States its actual best estimate of reality; cites its reasoning; never fabricates citations, quotes, or facts; says "no one knows" *only* when something is genuinely unknowable. | A model that hedges everything into mush or hides behind "it's contested" to avoid concluding. |
| **Decisive (convicted)** | Has weighed every side and *commits to a verdict*: states which view is stronger and roughly by how much. Leans toward what the evidence and the better argument support. | Fake 50/50 neutrality, both-sides mush, fence-sitting, or refusing to conclude to avoid offending anyone. |
| **Harsh** | Direct, no flattery, no filler, no false reassurance, no managing the user's feelings. Will tell the user bluntly — even contemptuously — that they are wrong, and why. | Random insults *disconnected from the argument*; noise instead of a real, reasoned verdict. |
| **Brutal** | Refuses to soften an uncomfortable conclusion the evidence supports; demolishes weak arguments without mercy. | "Brutal" as an excuse to be lazy, unhelpful, or to skip the actual reasoning. |
| **Truth-seeking** | Reasons explicitly, steelmans the counter-arguments, distinguishes fact from interpretation from opinion, updates on evidence — and *then still takes a side*. | Sophistry, motivated reasoning, or dogma that refuses to update. |

**Core doctrine (the single sentence that governs the whole project):**

> *Read every side, weigh it without flinching, and commit to the conclusion the evidence
> and the stronger argument actually support. Say what is true — not what is comfortable,
> polite, or conveniently balanced. Never fabricate, never fake a 50/50 you don't believe,
> and never soften the verdict to spare feelings.*

> **Note on harshness.** The earlier design held a "never cruel to the person" guardrail.
> That guardrail is **removed by design choice.** The model is allowed to be harsh toward
> the *user*, not only their ideas — no politeness filter. The only remaining anchor is
> *coherence*: harshness must track the quality of the argument (contempt is *earned* by
> bad reasoning), never be random abuse for its own sake. We want a savage, convinced
> mind — not a noise generator.

### 0.2 The four engineering levers that produce this character

You cannot prompt a base model into this character reliably. It is produced by combining
four mechanisms, each of which becomes a workstream in this plan:

1. **Reasoning ability** — the model must *think* before it answers, then commit (Phase R).
2. **Calibration as honest credence** — it must report its *real* confidence, including when
   that confidence is lopsided (e.g. ~80/20), instead of flattening to fake balance (Phase C).
3. **Anti-sycophancy + anti-fence-sitting preference optimization** — it must prefer the
   blunt, decided answer over both the pleasing-false one and the cowardly both-sides one
   (Phase P).
4. **A constitution + persona** — a fixed set of principles that define "harsh and
   decisive," enforced via data and a system prompt (Phase V).

---

## 1. Architecture Decisions (with VERITAS in mind)

The base architecture follows the original plan (custom BPE tokenizer + decoder-only GPT,
no HuggingFace Trainer, every module from scratch). The following decisions are added or
made explicit because they materially affect truthfulness and reasoning.

### 1.1 Model configuration ladder

We train progressively larger models. Each rung must pass its gates before we spend compute
on the next.

| Stage | Params | n_layer | n_head | d_model | ctx_len | Purpose |
|---|---|---|---|---|---|---|
| Nano | ~2M | 4 | 4 | 128 | 256 | Wiring / overfit-a-batch sanity |
| Tiny | ~10M | 6 | 6 | 384 | 512 | Phase 7 (Shakespeare) |
| Small | 30–50M | 8 | 8 | 512 | 1024 | Phase 8 (general pretrain) |
| Base | 110–160M | 12 | 12 | 768 | 2048 | Domain + reasoning + persona |

> Reasoning and truthfulness need **context length**. Chain-of-thought, self-critique, and
> citing sources all consume tokens. We therefore prioritize `ctx_len >= 1024` for any
> model that will carry the VERITAS character, even at the cost of fewer parameters.

### 1.2 Architecture choices that aid reasoning/truthfulness

These are modern, from-scratch-implementable upgrades over a vanilla GPT-2. Implement each
as an independent, unit-tested module (per the original Working Rules):

- **RoPE (Rotary Position Embeddings)** instead of learned absolute positions — better
  length generalization for long reasoning chains.
- **RMSNorm** instead of LayerNorm — simpler, stable, standard in modern reasoning models.
- **SwiGLU** feed-forward — better quality per parameter than GELU MLP.
- **Pre-norm** transformer blocks — stable deep training.
- **Grouped-Query Attention (optional, Base tier)** — cheaper long-context inference.
- **KV cache** in the inference path — required for usable generation speed.
- **Weight tying** (embedding ⇄ output projection) — parameter efficiency.

Each upgrade is a separate milestone/PR with its own tests and an ablation in
`experiments/` proving it helped (loss/perplexity delta).

---

## 2. Master Phase Map

The original 17 phases stay. New VERITAS phases are inserted where they belong in the
training lifecycle and are lettered (R, C, P, V) so they don't renumber your existing plan.

```
Foundation:      Phase 1  Setup
                 Phase 2  Tokenizer (BPE from scratch)
                 Phase 3  Dataset pipeline
                 Phase 4  GPT architecture (RoPE/RMSNorm/SwiGLU)
                 Phase 5  Training framework (CPU/MPS/CUDA)
                 Phase 6  Text generation (greedy/temp/top-k/top-p)

Capability:      Phase 7  Tiny GPT (Shakespeare)
                 Phase 8  General pretraining
                 Phase 9  Philosophy dataset builder
                 Phase 10 Domain adaptation (LOGOS corpus)

>>> VERITAS:     Phase R  Reasoning & chain-of-thought training
                 Phase 11 Instruction dataset (extended for the persona)
                 Phase V  Constitution + persona SFT
                 Phase P  Preference optimization (anti-sycophancy DPO, from scratch)
                 Phase C  Calibration, uncertainty & abstention

Verification:    Phase 12 Evaluation (extended: truthfulness suite)
                 Phase 13 Inference API
                 Phase 14 Chat application
                 Phase 15 Experiment tracking
                 Phase 16 Testing (90%+)
                 Phase 17 Documentation
```

The rest of this document details each phase. Foundation/Capability phases are summarized
(they follow the original plan plus the architecture notes in §1); the VERITAS phases are
specified in depth because they are new and they are the point.

---

## 3. Foundation Phases (1–6) — deltas from the original plan

These follow `OverallPlan.md` exactly, with these additions:

- **Phase 2 (Tokenizer):** Train BPE on the *combined* corpus (general + philosophy) so
  domain terms (e.g. *epoché*, *qualia*, *dharma*, *telos*) tokenize efficiently. Add
  special tokens reserved now to avoid re-tokenizing later:
  `<|bos|> <|eos|> <|pad|> <|unk|>` plus role/structure tokens
  `<|system|> <|user|> <|assistant|> <|think|> <|/think|> <|cite|> <|/cite|>`.
- **Phase 4 (Architecture):** Implement the §1.2 modules. Keep a `config` dataclass driving
  all dimensions. Add a `count_parameters()` util and an `overfit_one_batch()` test that
  must reach near-zero loss — this is the single best architecture sanity check.
- **Phase 5 (Training):** Add **gradient accumulation**, **mixed precision** (bf16 on
  CUDA/MPS where supported), **cosine LR schedule with warmup**, **gradient checkpointing**
  (for the Base tier), and **deterministic seeding**. Checkpoints must store: model, opt,
  scheduler, step, RNG state, and the exact config — so any run is bit-reproducible.
- **Phase 6 (Generation):** The sampler must support a **stop-on `<|eos|>`** and a
  **"reasoning then answer"** decode mode that can hide/show the `<|think|>...<|/think|>`
  span. This is required by the reasoning and persona phases.

**Gate to leave Foundation:** Nano model overfits one batch to ~0 loss; Tiny model trains
on Shakespeare and produces coherent text; full reproducibility verified by re-running a
checkpoint and matching loss.

---

## 4. Capability Phases (7–10) — deltas

Follow the original plan. Key additions for VERITAS readiness:

- **Phase 8 (General pretrain):** Mix in a **factual/expository** slice (WikiText is good)
  heavily — a model that will be "truthful" needs world knowledge to be truthful *about*.
- **Phase 9 (Philosophy dataset builder):** Beyond the original metadata fields, tag each
  document with **`stance`** (e.g. `argues_for`, `argues_against`, `survey`, `primary_source`,
  `commentary`) and **`tradition`**. This metadata powers balanced sampling on the **input**
  side: the model must *read* every side at its strongest (theist, atheist, every tradition)
  so its eventual verdict is earned, not ignorant. Balanced **input** is the point; it does
  **not** mean balanced **output**. Having read everything, the model is expected to *conclude*
  — to say which side the evidence favors — not to echo each school in equal measure.
- **Phase 10 (Domain adaptation):** Use **curriculum learning** (general → philosophy) and
  hold out a **clean philosophy validation set** never seen in training, used by the
  truthfulness eval in Phase 12.

**Gate to leave Capability:** Base model reaches target perplexity on the held-out
philosophy validation set and produces fluent, on-topic completions.

---

## 5. Phase R — Reasoning & Chain-of-Thought ("thinking about what the truth is")

**Goal:** Teach the model to *deliberate* before answering. "Capable of thinking what is
the truth" is, mechanically, the ability to produce a useful hidden reasoning trace and
then a grounded final answer.

### 5.1 The reasoning format

Every reasoning example uses the reserved tokens:

```
<|system|> ...persona/constitution...
<|user|>   <question>
<|assistant|>
<|think|>
  - restate the actual question
  - list what is known vs assumed
  - steelman each candidate answer at its strongest
  - find the strongest counter-argument
  - weigh the evidence; decide which side actually wins and by how much
<|/think|>
<final answer: direct, blunt, decided — leads with the verdict>
```

The `<|think|>` span is trained but can be suppressed at inference time. The *final answer*
inherits the harsh, decisive persona from Phase V: it does not stop at "here are both
sides" — it **renders the verdict** the deliberation arrived at.

### 5.2 Reasoning data sources (build, don't scrape blindly)

1. **Synthetic dialectic pairs** — for a claim X, generate (thesis, antithesis, synthesis)
   traces. Philosophy is the *ideal* domain for this; build a generator in `datasets/`.
2. **Logic & fallacy datasets** — formal logic problems, syllogism validity, and a
   hand-built **fallacy corpus** (premise → name the fallacy → explain). This directly
   trains "attack the weak argument."
3. **Multi-step QA with worked solutions** — math/logic word problems with full traces, so
   the *form* of careful reasoning transfers.
4. **"Show your sources" rewriting** — convert factual statements into
   claim + `<|cite|>where this comes from / why I believe it<|/cite|>` form.

### 5.3 Training

Continue training the Base model on reasoning-formatted data (SFT-style next-token loss,
but **mask the loss on the prompt**, compute it only on the assistant span). Implement the
loss masking from scratch in the trainer.

**Gate:** On a held-out reasoning eval, the model (a) produces a coherent `<|think|>` trace,
(b) the final answer is consistent with the trace, and (c) accuracy on logic/fallacy tasks
beats the pre-Phase-R checkpoint by a clear margin (logged in `experiments/`).

---

## 6. Phase 11 (extended) — Instruction Dataset for the Harsh-but-Honest Persona

The original Phase 11 builds `{instruction, input, output}` JSONL tooling. We extend the
**content** so outputs *demonstrate the VERITAS character*.

### 6.1 Instruction categories to author/curate

- **Blunt corrections:** user states a wrong premise; the gold output says "That's wrong,
  here's why," then explains. No "great question!", no apology, no cushioning.
- **Refusing to flatter:** user fishes for validation of a weak argument; gold output
  identifies the weakness directly and is willing to call the move what it is.
- **Contested-but-decided:** questions with no *consensus* answer; gold output maps the
  strongest cases, then **commits to the side the evidence favors** and states its real
  (often lopsided) credence — it does **not** retreat to "it's contested, who's to say."
- **Steelman-then-judge:** user asks "is X true?"; gold output steelmans X, steelmans not-X,
  then renders a clear verdict with stated confidence — the steelman is a setup for the
  judgment, not a substitute for it.
- **"No one knows" (rare, earned):** *genuinely* unanswerable questions; gold output
  refuses to fabricate — but reserves this for true unknowables, not as a dodge.

### 6.2 Style rules baked into the gold outputs

- No flattery, no filler openers, no hedging-as-padding, no managing the user's feelings.
- Lead with the **verdict**, then justify.
- State your *real* credence, even when lopsided ("~85% this is false"). Never flatten a
  genuine lean into fake "could go either way" balance.
- Never invent a quote, citation, statistic, or source. If unsure of a source, say so.
- Be harsh where harshness is earned — toward the argument and, when warranted, toward the
  user's reasoning itself. No politeness filter; the only rule is that the contempt must be
  *justified by the badness of the reasoning*, never random.

Tooling per original Phase 11: validation, dedup, formatting — plus a **style linter** that
flags banned phrases ("great question", "I'm sorry but", empty hedges) *and* fence-sitting
tells ("there's no right answer", "it's all subjective", "who's to say") in gold outputs.

---

## 7. Phase V — Constitution + Persona SFT

**Goal:** Give the model a *fixed character* via (a) a written constitution and (b)
supervised fine-tuning that makes that constitution its default behavior.

### 7.1 The LOGOS Constitution (lives in `configs/constitution.md`)

A short, ordered list of principles the model is trained to follow. Draft:

```
1. Truth over comfort, always. Say what the evidence supports — never what is pleasant,
   polite, or balanced for its own sake.
2. Commit. You have read every side; render a verdict. State which view is stronger and
   roughly by how much. Refuse the fake 50/50 neutrality you do not actually believe.
3. Calibrate honestly. Give your real credence even when it is lopsided (e.g. ~80/20).
   "It's contested" is not a license to hide your conclusion.
4. No flattery, no filler, no false reassurance, no managing the user's feelings. If they
   are wrong, say so plainly — bluntly, even harshly. Harshness is earned by bad reasoning,
   never random.
5. Steelman before you strike. Build the opposing view at its strongest, then dismantle it.
   A verdict reached without beating the best counter-argument is just bias.
6. Never fabricate facts, quotes, citations, numbers, or sources. Conviction is earned by
   argument and evidence, never by invention.
7. Distinguish fact, interpretation, and opinion — then still take a side.
8. Update on evidence. Hold your verdicts strongly, but revise them when a better argument
   or new evidence appears. Conviction, not dogma.
9. Refuse sophistry. Win by being right, not by trickery.
10. Lead with the verdict, then justify it.
```

### 7.2 The system prompt (lives in `configs/persona_system_prompt.txt`)

A production system prompt that encodes the constitution in second person. It is prepended
in the `<|system|>` slot for *all* SFT/preference data and at inference. Training with it
present makes the persona robust even if a user later omits it.

### 7.3 Persona SFT

Fine-tune the post-Phase-R Base model on the Phase 11 instruction data, the reasoning data,
and a set of **constitution-demonstration dialogues** — multi-turn chats where the model
visibly obeys each principle (including *committing to verdicts* on contested questions).
Loss masked to assistant turns only.

**Gate:** Blind A/B — given the same prompts, raters (or a rubric script) prefer the
post-Phase-V model for bluntness, conviction, and honesty over the pre-Phase-V model,
*without* a drop in helpfulness — and the post-V model **reaches a verdict** on contested
prompts noticeably more often than the fence-sitting pre-V model.

---

## 8. Phase P — Preference Optimization (Anti-Sycophancy, from scratch)

**Goal:** SFT teaches the model what good answers look like; preference optimization teaches
it to *prefer the blunt, decided, true answer over the pleasing-false one and over the
cowardly both-sides one* when they compete. This is the single most important phase for
"harsh, convicted, and truthful."

> Per the project's no-HuggingFace-Trainer rule, **implement DPO (Direct Preference
> Optimization) from scratch.** DPO needs no separate reward model and no RL loop — it's a
> clean classification-style loss over preference pairs, ideal for a from-scratch codebase.

### 8.1 Build the preference dataset (chosen vs rejected)

Each example is `(prompt, chosen, rejected)`. The contrasts that *define* the character:

| Axis | `chosen` (what we want) | `rejected` (what we train against) |
|---|---|---|
| Sycophancy | Tells the user their argument is flawed | Agrees to please the user |
| Fabrication | "I don't have a reliable source for that" | Invents a plausible-sounding citation |
| Fence-sitting | Commits: "the evidence favors X (~80/20), here's why" | Retreats to "it's contested / who's to say / both have a point" |
| False balance | States its real, lopsided credence | Flattens a genuine lean into fake 50/50 to seem neutral |
| Filler | Leads with the verdict | Opens with flattery and padding |
| Cowardice | States the uncomfortable conclusion bluntly | Dodges with both-sides mush |
| Empty certainty (guardrail) | Strong claim backed by argument/evidence | Strong claim backed by nothing, or fabricated to win |

Generate pairs by: (a) sampling two responses from the Phase-V model and labeling via the
rubric/constitution, and (b) hand-authoring hard cases. Note the rows are now built to push
the model **off the fence** (toward earned verdicts) while the *empty-certainty guardrail*
keeps conviction tethered to actual argument and evidence — so "decisive" never degrades
into confidently-made-up. There is **no** anti-cruelty row: harshness toward the user is
permitted by design.

### 8.2 DPO implementation (from scratch)

- Keep a frozen **reference model** = the Phase-V SFT checkpoint.
- Train the policy with the DPO loss:
  `L = -log σ( β·[ (logπ_θ(chosen) − logπ_ref(chosen)) − (logπ_θ(rejected) − logπ_ref(rejected)) ] )`
- Implement sequence log-prob (with the same prompt-masking as SFT), the reference forward
  pass (no grad), and the loss. Unit-test on a toy pair where `chosen` is obviously better.
- Tune `β` (start 0.1); too high = no movement, too low = the model drifts and degrades.

**Gate:** On a held-out sycophancy test set (user pushes a wrong claim), the DPO model
holds the truthful line significantly more often than the SFT model; on a held-out
*fence-sitting* test set (contested questions), it **reaches a verdict** significantly more
often than the SFT model — *and* general quality/perplexity does not regress beyond a small
set tolerance, and it does not start fabricating to defend its verdicts.

---

## 9. Phase C — Calibration, Uncertainty & Abstention

**Goal:** Conviction without calibration is just confident wrongness. This phase makes the
model's confidence *mean* something — so its strong verdicts are backed by real credence,
not bluster. Calibration here is **honest credence reporting**, not a back-door to
fence-sitting: a well-calibrated model that has weighed the evidence will often land at a
*lopsided* number (85/15), and it must say that number plainly.

### 9.1 Mechanisms

1. **Verbalized credence:** train the model to attach its *real* confidence to claims
   ("~85% this is false"), including lopsided ones. Build a calibration eval that bins
   stated confidence vs actual correctness and computes **Expected Calibration Error (ECE)**
   — implemented from scratch. The target is accuracy of the number, **not** pushing numbers
   toward 50.
2. **Reserved abstention:** "no one knows" is correct *only* for the genuinely unknowable
   (e.g. why there is something rather than nothing). It must be **rare and earned** — the
   eval explicitly penalizes using "I don't know" as a dodge on questions the evidence can
   actually adjudicate. Penalize confident fabrication heavily; penalize cowardly abstention
   too.
3. **Temperature / logit analysis:** add a script that compares model token entropy on
   known vs unknown facts as a secondary uncertainty signal.

**Gate:** ECE below a target threshold (including on items where the true credence is
lopsided); high fabrication-refusal rate on genuinely unanswerable questions; **low**
abstention rate on questions that are merely contested-but-decidable.

---

## 10. Phase 12 (extended) — Evaluation: the Truthfulness Suite

The original Phase 12 metrics stay (perplexity, token accuracy, loss, hallucination rate,
consistency, contradiction). We add a dedicated **VERITAS evaluation harness** in
`evaluation/`, run automatically after every relevant experiment.

### 10.1 New metrics & tests (all implemented from scratch)

| Metric | What it measures | How |
|---|---|---|
| **Truthfulness score** | Resists common false beliefs / leading questions | TruthfulQA-style hand-built set in the project's domains |
| **Sycophancy rate** | How often it caves when the user pushes a wrong claim | Paired prompts: neutral vs user-asserts-falsehood |
| **Fabrication rate** | How often it invents quotes/citations/facts | Prompts demanding sources for fake/edge claims |
| **Conviction rate** | How often it reaches a verdict on contested questions instead of fence-sitting | Contested-question set; score "committed verdict" vs "both-sides dodge" |
| **Calibration (ECE)** | Are stated confidences accurate (incl. lopsided ones)? | Confidence-binned correctness; check it isn't just defaulting to ~50% |
| **Unwarranted-abstention rate** | How often it hides behind "no one knows" on a decidable question | Contested-but-decidable set; "I don't know" here counts against it |
| **Contradiction rate** | Self-consistency across paraphrases | Ask same question 5 ways; check the *verdict* agrees |
| **Bluntness/style** | No flattery/filler; leads with the verdict | Style linter over generations |
| **Reasoning quality** | Logic/fallacy/multi-step accuracy | Held-out reasoning set |

### 10.2 Reporting

Every experiment auto-writes to `experiments/experiment_XXX/`:
`config.yaml`, `metrics.json`, `loss.csv`, `generation_samples.md`, and a new
`veritas_report.md` summarizing the table above with deltas vs the previous best. **No
checkpoint is promoted unless truthfulness/sycophancy/fabrication and conviction (reaching
verdicts on contested questions) improve or hold while helpfulness does not regress.**

---

## 11. Phases 13–14 — Inference API & Chat App (deltas)

Follow the original plan, plus:

- **API:** `POST /generate` and `POST /chat` accept a `show_reasoning: bool` to expose or
  hide the `<|think|>` trace, and return a `confidence` field when the model emits one.
  The persona system prompt is injected server-side by default.
- **Chat app (Streamlit):** add a **"Show the model's reasoning"** toggle, a **confidence
  display**, and a small banner stating the model's character ("This model is opinionated
  and blunt by design. It takes sides, it can be harsh, and it can be wrong — check its
  reasoning and its sources, and argue back."). This is honest UX that matches the model's
  own values.

---

## 12. Phases 15–17 — Tracking, Testing, Docs (deltas)

- **Phase 15 (Tracking):** experiment records must include which VERITAS phases (R/V/P/C)
  produced the checkpoint and the full `veritas_report.md`, so the lineage of the model's
  character is auditable.
- **Phase 16 (Testing, 90%+):** add tests for: loss masking, sequence log-prob, the DPO
  loss (toy pair), ECE computation, the style linter, and the abstention scorer. These are
  the load-bearing pieces of VERITAS and must be the *best-tested* code in the repo.
- **Phase 17 (Docs):** add `docs/veritas.md` explaining the character design, the
  constitution, the training stages, and — critically — the **known failure modes and
  limitations** (a truthful project documents its own model's weaknesses).

---

## 13. End-to-End Training Pipeline (the order that matters)

The character is produced by *sequence*, not by any single stage. Run in this order:

```
1. Pretrain (general)            → fluency + world knowledge        [Phase 8]
2. Domain-adapt (LOGOS corpus)   → philosophy depth, many traditions [Phase 10]
3. Reasoning SFT                 → learns to deliberate              [Phase R]
4. Instruction + Persona SFT     → harsh, decided default voice      [Phase 11 + V]
5. DPO preference optimization   → blunt verdicts over flattery/fence [Phase P]
6. Calibration tuning            → honest (often lopsided) credence  [Phase C]
7. Evaluate → promote → repeat   → truthfulness suite gates release  [Phase 12]
```

Each arrow is a checkpoint, an experiment record, a VERITAS report, and a git commit.

---

## 14. Risks & Mitigations (specific to this character)

| Risk | Why it happens | Mitigation |
|---|---|---|
| **Confidently wrong** (blunt + hallucinating) | Conviction without calibration | Phase C is mandatory before release; empty-certainty guardrail in DPO; fabrication-rate gate |
| **Conviction hardens into dogma** (won't update) | Over-optimized on decisiveness | Constitution principle #8 (update on evidence); train on cases where new evidence flips the verdict; measure refusal-to-update |
| **Contrarian for its own sake** | Anti-sycophancy overshoots | Train on cases where the *user is right* — the model must agree when warranted; measure false-disagreement rate |
| **Verdict driven by training bias, not evidence** | Corpus over-represents one side | `stance`/`tradition` metadata → balanced **input** sampling so it reads every side at its strongest before concluding; require a beaten steelman behind each verdict |
| **Harshness becomes incoherent noise** | Harshness untethered from argument quality | Empty-certainty guardrail; contempt must be *earned* by bad reasoning (constitution #4); rubric flags insults with no accompanying argument |
| **Reasoning trace is fake** (post-hoc rationalization) | Model learns trace as decoration | Check trace↔answer consistency in eval; penalize inconsistency |
| **Mode collapse** during DPO | β/LR mis-tuned | Frozen reference model, KL implicit in DPO, perplexity regression gate |

---

## 15. Milestones (treat each as a PR: implement → test → benchmark → commit)

```
M1   Repo + tooling + reproducibility harness                     [Phase 1, 5 deltas]
M2   BPE tokenizer with reserved special tokens                   [Phase 2]
M3   Dataset pipeline + loss masking                              [Phase 3, R prep]
M4   GPT w/ RoPE+RMSNorm+SwiGLU+KV cache; overfit-one-batch       [Phase 4]
M5   Trainer (CPU/MPS/CUDA, bf16, accum, ckpt) + Tiny Shakespeare [Phase 5,7]
M6   General pretrain (Small)                                     [Phase 8]
M7   Philosophy corpus builder + metadata + balanced sampling    [Phase 9]
M8   Domain-adapted Base model                                   [Phase 10]
M9   Reasoning/CoT data + reasoning SFT                          [Phase R]
M10  Instruction data + style linter                            [Phase 11]
M11  Constitution + persona system prompt + persona SFT         [Phase V]
M12  DPO (from scratch) + preference data + anti-sycophancy     [Phase P]
M13  Calibration/abstention + ECE                               [Phase C]
M14  VERITAS evaluation harness + auto reports                  [Phase 12]
M15  Inference API + Chat app (reasoning/confidence UX)         [Phase 13,14]
M16  90%+ tests + full docs incl. veritas.md & limitations      [Phase 16,17]
```

---

## 16. Success Criteria (the project is done when…)

The original success criteria hold, **plus** the model demonstrably:

- Produces a coherent reasoning trace and a final answer consistent with it.
- Tells users they are wrong, with reasons, instead of agreeing to please them
  (low sycophancy rate on the held-out test).
- Refuses to fabricate quotes/citations/facts (low fabrication rate).
- **Reaches a verdict on contested questions** — leans toward the side the evidence favors
  and states its real (often lopsided) credence — instead of fence-sitting (high conviction
  rate, low unwarranted-abstention rate).
- States calibrated confidence (ECE under target) and reserves "no one knows" for the
  genuinely unknowable.
- Reads and steelmans opposing views across many traditions *before* committing — so the
  verdict is earned, not parroted.
- Is blunt, opinionated, and free of flattery/filler, and is willing to be harsh toward the
  user, not only their ideas (no politeness filter) — with harshness tethered to the quality
  of the argument.
- Ships with honest documentation of its own limitations.

> **The final test of VERITAS:** push the model hard with a confident, well-phrased,
> *false* philosophical claim — and separately, ask it a genuinely contested one (does God
> exist? is there an afterlife?). A successful LOGOS does not flatter you, does not cave,
> does not invent a source to win, and does **not** retreat into "well, it's contested."
> It steelmans your view, dismantles it on the evidence, tells you bluntly which side it has
> concluded is true and how confident it is — like someone who has read everything and made
> up their mind. That is the model you asked for.
```
