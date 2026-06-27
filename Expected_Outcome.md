# Project LOGOS — Expected Outcome After Each Phase

> Companion to `Detailed_Implementation_Plan.md`. For every phase, this file states the
> **concrete artifact**, the **observable behavior**, the **pass/fail gate**, and the
> **failure signal** to watch for. A phase is "done" only when its gate is green — never
> move on because code merely runs.

How to read each entry:

- **Artifacts** — files/checkpoints that must exist when the phase ends.
- **Observable outcome** — what you can actually see/run and verify.
- **Pass gate (must hold)** — the measurable bar to advance.
- **Red flag** — the most common way this phase silently fails.

---

## Foundation

### Phase 1 — Project Setup
- **Artifacts:** full repo tree (`model/ tokenizer/ training/ ...`), `pyproject.toml`,
  `requirements.txt`, `Makefile`, `.gitignore`, `README.md`.
- **Observable outcome:** fresh clone + `make install` succeeds; `pytest` runs (even if
  empty); imports resolve.
- **Pass gate:** environment reproduces on a clean machine with one command.
- **Red flag:** "works on my machine" — undocumented manual setup steps.

### Phase 2 — Tokenizer (BPE from scratch)
- **Artifacts:** trained vocab/merges files; all reserved special tokens registered
  (`<|bos|> <|eos|> <|pad|> <|unk|> <|system|> <|user|> <|assistant|> <|think|> <|/think|>
  <|cite|> <|/cite|>`).
- **Observable outcome:** `decode(encode(text)) == text` for arbitrary input including
  domain terms (*epoché, qualia, dharma, telos*).
- **Pass gate:** lossless round-trip on a held-out text; reasonable compression ratio;
  domain terms don't explode into single bytes.
- **Red flag:** round-trip fails on whitespace/unicode; special tokens get split.

### Phase 3 — Dataset Pipeline
- **Artifacts:** dataset classes, chunker, batch collator, loss-masking utility.
- **Observable outcome:** a batch yields correct `(input_ids, targets, loss_mask)` shapes;
  prompt tokens are masked, only completion contributes to loss.
- **Pass gate:** unit test proves loss mask zeros out prompt positions; deterministic
  sampling with a fixed seed.
- **Red flag:** off-by-one between inputs and targets; mask applied to wrong span.

### Phase 4 — GPT Architecture (RoPE / RMSNorm / SwiGLU / KV cache)
- **Artifacts:** independent, tested modules; a `config` dataclass; `count_parameters()`.
- **Observable outcome:** **Nano model overfits a single batch to ~0 loss.**
- **Pass gate:** overfit-one-batch test passes; each architecture module has its own unit
  test; forward/backward shapes correct with and without KV cache.
- **Red flag:** can't drive loss to ~0 on one batch → a wiring/masking bug exists. Stop and
  fix before anything else.

### Phase 5 — Training Framework (CPU / MPS / CUDA)
- **Artifacts:** trainer with grad accumulation, mixed precision, cosine LR + warmup, grad
  clipping, checkpoint save/resume (model+opt+scheduler+step+RNG+config).
- **Observable outcome:** training loss decreases smoothly; resume-from-checkpoint matches
  the pre-save loss curve.
- **Pass gate:** **bit-reproducible** — re-running a checkpoint reproduces the loss.
- **Red flag:** loss spikes/NaNs (LR or precision issue); resumed run diverges from original.

### Phase 6 — Text Generation
- **Artifacts:** sampler (greedy, temperature, top-k, top-p), stop-on-`<|eos|>`,
  show/hide `<|think|>` mode.
- **Observable outcome:** `generate.py` produces coherent text and stops cleanly at EOS.
- **Pass gate:** each sampling strategy works; KV-cache generation matches no-cache output
  for greedy decoding.
- **Red flag:** runaway generation that never stops; cached vs uncached mismatch.

**Foundation exit:** Nano overfits one batch; Tiny trains; full reproducibility verified.

---

## Capability

### Phase 7 — Tiny GPT (Shakespeare)
- **Artifacts:** ~10M-param Tiny checkpoint; sample outputs in `experiments/`.
- **Observable outcome:** model generates fluent Shakespeare-style English.
- **Pass gate:** validation loss/perplexity hits target; samples are clearly English, not
  gibberish.
- **Red flag:** memorization (verbatim training text) instead of style generalization.

### Phase 8 — General Pretraining
- **Artifacts:** 30–50M Small checkpoint trained on TinyStories/WikiText/OpenWebText subset.
- **Observable outcome:** coherent general-purpose completions; basic world knowledge.
- **Pass gate:** target perplexity on a general held-out set; factual/expository slice well
  represented.
- **Red flag:** fluent but factually empty — too little expository data to be "truthful about."

### Phase 9 — Philosophy Dataset Builder
- **Artifacts:** ingest/clean/dedup tools; JSONL corpus with metadata incl.
  `title, author, year, school, religion, topic, language` **plus `stance` and `tradition`**.
- **Observable outcome:** corpus is queryable; balanced sampler can draw multiple sides of
  a question.
- **Pass gate:** dedup verified; metadata coverage high; multiple traditions represented.
- **Red flag:** corpus skewed to one school → future "truth-seeking" is just bias.

### Phase 10 — Domain Adaptation (LOGOS Corpus)
- **Artifacts:** ~110–160M domain-adapted Base checkpoint; clean held-out philosophy
  validation set (never trained on).
- **Observable outcome:** fluent, on-topic philosophy/religion/ethics completions.
- **Pass gate:** target perplexity on the held-out philosophy validation set; curriculum
  (general→domain) applied.
- **Red flag:** catastrophic forgetting of general fluency; overfit to corpus phrasing.

**Capability exit:** Base model hits philosophy-validation perplexity and stays fluent.

---

## VERITAS (the character)

### Phase R — Reasoning & Chain-of-Thought
- **Artifacts:** reasoning-formatted dataset (`<|think|>...<|/think|>` + final answer);
  fallacy/logic corpus; reasoning-SFT checkpoint.
- **Observable outcome:** model produces a real deliberation trace that *weighs the sides
  and lands on a verdict*, then a final answer consistent with it; can name logical fallacies.
- **Pass gate:** logic/fallacy accuracy beats the pre-R checkpoint by a clear margin;
  trace↔answer consistency high; the trace ends in a decision, not a shrug.
- **Red flag:** the `<|think|>` trace is decorative — answer contradicts its own reasoning,
  or the deliberation ends in "both sides have a point" with no verdict.

### Phase 11 — Instruction Dataset (extended persona)
- **Artifacts:** `{instruction, input, output}` JSONL; **style linter** flagging flattery/
  filler **and fence-sitting tells**; categories: blunt corrections, refusing flattery,
  contested-but-decided, steelman-then-judge, "no one knows" (rare/earned).
- **Observable outcome:** gold outputs lead with the verdict, no "great question!", no
  invented sources, no "it's all subjective" dodges.
- **Pass gate:** style linter passes on the dataset; dedup + format validation green.
- **Red flag:** gold data itself is sycophantic, hedgy, or fence-sitting — the model copies it.

### Phase V — Constitution + Persona SFT
- **Artifacts:** `configs/constitution.md`, `configs/persona_system_prompt.txt`,
  persona-SFT checkpoint.
- **Observable outcome:** default voice is blunt, opinionated, and decisive; obeys the 10
  constitution principles even when the system prompt is omitted; commits to verdicts on
  contested questions.
- **Pass gate:** blind A/B — raters/rubric prefer post-V for bluntness, conviction, and
  honesty **with no helpfulness drop**; post-V reaches verdicts on contested prompts more
  often than the fence-sitting pre-V model.
- **Red flag:** persona only appears with the system prompt present (not internalized), or
  the model stays neutral/wishy-washy on questions it should decide.

### Phase P — Preference Optimization (Anti-Sycophancy + Anti-Fence-Sitting DPO, from scratch)
- **Artifacts:** preference dataset `(prompt, chosen, rejected)` covering sycophancy,
  fabrication, fence-sitting, false balance, filler, cowardice, **and the empty-certainty
  guardrail**; frozen reference model; DPO checkpoint.
- **Observable outcome:** when the user pushes a confident *wrong* claim, the model holds the
  truthful line; on a contested question, it *commits to a verdict* instead of hedging.
- **Pass gate:** sycophancy rate drops sharply and conviction rate rises sharply vs the SFT
  model **and** perplexity/quality does not regress beyond tolerance; the model does not start
  fabricating to defend its verdicts (empty-certainty guardrail holds).
- **Red flag:** mode collapse (degenerate outputs from bad β/LR), the model becomes
  contrarian even when the user is right, or it commits confidently to things it made up.

### Phase C — Calibration & Earned Credence
- **Artifacts:** calibration eval with from-scratch **ECE**; contested-but-decidable set;
  small genuinely-unanswerable set.
- **Observable outcome:** model attaches its *real* confidence to claims (including lopsided
  ones like ~85%), and reserves "no one knows" for the genuinely unanswerable.
- **Pass gate:** ECE below target threshold (including on lopsided items); high
  fabrication-refusal on true unknowables; **low** abstention on merely-contested questions.
- **Red flag:** blunt **and wrong** (confident on things it can't know), or calibration
  collapsing into fake ~50% balance to dodge committing.

---

## Verification

### Phase 12 — Evaluation (Truthfulness Suite)
- **Artifacts:** `evaluation/` harness; per-experiment `veritas_report.md` with truthfulness,
  sycophancy, fabrication, conviction rate, ECE, unwarranted-abstention, contradiction,
  bluntness, reasoning metrics.
- **Observable outcome:** every experiment auto-emits a report with deltas vs previous best.
- **Pass gate:** **no checkpoint is promoted** unless truthfulness/sycophancy/fabrication and
  conviction improve or hold while helpfulness does not regress.
- **Red flag:** cherry-picked samples instead of the full metric table.

### Phase 13 — Inference API (FastAPI)
- **Artifacts:** `POST /generate`, `POST /chat` (with `show_reasoning`, returns
  `confidence`), `GET /health`, `GET /model`; server-side persona injection.
- **Observable outcome:** API serves the model; reasoning trace toggle and confidence field
  work end-to-end.
- **Pass gate:** endpoints tested; persona applied by default; latency acceptable with KV
  cache.
- **Red flag:** persona missing unless the client supplies it.

### Phase 14 — Chat Application (Streamlit)
- **Artifacts:** chat UI with reasoning toggle, confidence display, honesty banner, params,
  history, model selection.
- **Observable outcome:** a usable conversation; user can see how the model reasoned and how
  confident it is.
- **Pass gate:** end-to-end chat works against the API; UX matches the model's stated values.
- **Red flag:** UI hides uncertainty/reasoning — dishonest UX for an honesty-focused model.

### Phase 15 — Experiment Tracking
- **Artifacts:** auto-created `experiments/experiment_XXX/` (`config.yaml`, `metrics.json`,
  `loss.csv`, `generation_samples.md`, `veritas_report.md`) tagged with which R/V/P/C stages
  produced the checkpoint.
- **Observable outcome:** any model's character lineage is fully auditable.
- **Pass gate:** every training run produces a complete, reproducible record.
- **Red flag:** results exist but the config/lineage that produced them is lost.

### Phase 16 — Testing (90%+ coverage)
- **Artifacts:** test suite incl. loss masking, sequence log-prob, DPO loss (toy pair), ECE,
  style linter, abstention scorer.
- **Observable outcome:** `pytest` green; coverage report ≥ 90%.
- **Pass gate:** the load-bearing VERITAS pieces are the best-tested code in the repo.
- **Red flag:** high coverage of trivial code, low coverage of DPO/calibration/masking.

### Phase 17 — Documentation
- **Artifacts:** README, architecture diagrams, API/training/inference/developer guides,
  and **`docs/veritas.md`** including known failure modes & limitations.
- **Observable outcome:** a new engineer can reproduce training and understand the character
  design from docs alone.
- **Pass gate:** docs let someone reproduce a run and explain *why* the model behaves as it
  does — limitations honestly stated.
- **Red flag:** docs describe the architecture but omit the model's weaknesses (un-truthful
  about a truthfulness project).

---

## Final Expected Outcome (whole project)

When all phases pass their gates, you have a from-scratch GPT specialized in philosophy/
religion/ethics/psychology/logic that behaves like someone who has *read every side and made
up their mind.* On the **final VERITAS test** — pushed with a confident, well-phrased,
*false* claim, and separately asked a genuinely contested one (does God exist? is there an
afterlife?) — it will:

1. **Steelman** the opposing view at its strongest.
2. **Dismantle** it on the evidence, leading with the verdict.
3. **Commit to a side** — say which view it has concluded is more true and lean into it —
   instead of retreating to "it's contested."
4. **Refuse to fabricate** a source to win.
5. **State its real confidence** (often lopsided) and reserve "no one knows" for the
   genuinely unknowable.
6. Be **blunt and harsh** — toward your reasoning and, when earned, toward you — with the
   harshness tethered to the quality of the argument, never random.

That observable behavior — reproducibly, with the metrics in `veritas_report.md` to prove
it — is the definition of done.
