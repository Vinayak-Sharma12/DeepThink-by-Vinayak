# AGENTS.md — Build Rules for Project LOGOS

> Model display name: **DeepThink by Vinayak** · internal codename: **LOGOS** · character
> pillar: **VERITAS**.

> Persistent instructions for any AI agent (Cursor, etc.) working in this repo. Read this
> before writing code. The full spec is in `Technical_Spec.md`; the plan is in
> `Detailed_Implementation_Plan.md`; the phase gates are in `Expected_Outcome.md`.

## What we're building

A GPT-style language model **from scratch in PyTorch**, specialized in philosophy, religion,
spirituality, psychology, ethics, and logic. Its character ("VERITAS") is **truthful, blunt,
opinionated, and decisive** — it reads every side, commits to a verdict, states its real
(often lopsided) confidence, and is allowed to be harsh toward the user, not just their ideas.

## Hard rules (never violate)

- **From scratch only.** No HuggingFace `Trainer`. No copied GPT implementations. Implement
  every major component in-repo (tokenizer, attention, norms, FFN, trainer, DPO, ECE).
- **Preference optimization is DPO**, implemented from scratch. No reward model, no RL/PPO loop.
- **Every module is independent and unit-tested.** No module lands without tests.
- **Type hints everywhere. Docstrings on every public function. PEP8.**
- **Reproducibility is mandatory.** Seed RNG; checkpoints store model+optimizer+scheduler+
  step+RNG state+`GPTConfig`. Resume must reproduce the loss curve.
- **No TODOs** left in committed code unless explicitly requested.
- **Don't fabricate.** No invented benchmarks, citations, or results in code, comments, or docs.

## Architecture defaults (see `Technical_Spec.md` §4)

- Decoder-only, pre-norm. **RoPE**, **RMSNorm**, **SwiGLU**, causal attention (**GQA** optional
  at Base+), **KV cache**, **weight tying**.
- All dimensions flow from a single `GPTConfig` dataclass. Never hardcode dims in modules.
- Size ladder: Nano → Tiny → Small → **Medium (~75M)** → Base. Each rung must pass its gate
  before spending compute on the next.

## Workflow (every feature)

1. Design → 2. Implement → 3. Unit test → 4. Integration test → 5. Benchmark →
6. Document → 7. Git commit. **Never skip a step.** Treat each milestone as a PR.

- The **overfit-one-batch** test (Nano → ~0 loss) is the first gate for any architecture change.
- KV-cache decoding **must** match no-cache greedy decoding (parity test).
- Loss masking: prompt tokens masked (loss = 0), only completion contributes. Watch the
  off-by-one between `input_ids` and `targets`.

## The VERITAS character (when generating data / gold outputs / persona)

- Lead with the **verdict**, then justify. No "great question!", no flattery, no filler.
- **Commit** on contested questions. Do not fence-sit ("it's contested / who's to say").
  Reserve "no one knows" for the genuinely unanswerable.
- State **real, often lopsided** confidence (e.g. "~85%"). Never fake a 50/50.
- **Steelman before you strike.** Never **fabricate** facts, quotes, citations, or numbers.
- Harshness is allowed toward the user, but must be **earned by bad reasoning** — never random
  abuse. (The old "never cruel to the person" guardrail is intentionally removed; the
  coherence anchor replaces it.)
- The canonical voice is defined by `Sample_Answers.md` and `Model_Answers_50.md`. Match it.

## Alignment-critical code (must be the best-tested in the repo)

`loss masking`, `sequence log-prob`, `DPO loss`, `ECE`, `style linter`, `abstention scorer`.

## Don't

- Don't add dependencies casually; prefer stdlib + PyTorch. Justify any new dependency.
- Don't promote a checkpoint unless the VERITAS report shows truthfulness/sycophancy/
  fabrication/conviction improve or hold while helpfulness does not regress.
- Don't change `vocab_size` after the tokenizer is trained (it forces a re-tokenization).
