# Phase Implementation Plans — Index

> Per-phase, Cursor-ready implementation plans for **DeepThink by Vinayak** (codename LOGOS,
> character pillar VERITAS). Each file is self-contained: hand it to Cursor and it has the
> objective, prerequisites, exact files to create, function signatures, required tests, the
> acceptance gate, and a kickoff prompt.

## How to use these with Cursor

1. Work **one phase per file, in order.** Do not start a phase until its prerequisites pass.
2. Treat each phase as a **pull request**: implement → unit test → integration test →
   benchmark → document → commit.
3. The **kickoff prompt** at the bottom of each file is what you paste into Cursor to start.
4. A phase is "done" only when its **acceptance gate** (from `Expected_Outcome.md`) is green.
5. `AGENTS.md` rules always apply (from scratch, typed, tested, reproducible, no HF Trainer).

## Reading order (pipeline order)

| # | File | Phase | Gate (short) |
|---|---|---|---|
| 01 | `01_phase1_project_setup.md` | Project setup | clean clone + `make install` + `pytest` works |
| 02 | `02_phase2_tokenizer.md` | BPE tokenizer | lossless round-trip; special tokens intact |
| 03 | `03_phase3_dataset_pipeline.md` | Dataset pipeline | loss mask zeroes prompt; shapes correct |
| 04 | `04_phase4_gpt_architecture.md` | GPT architecture | overfit-one-batch → ~0 loss |
| 05 | `05_phase5_training_framework.md` | Training framework | bit-reproducible resume |
| 06 | `06_phase6_text_generation.md` | Generation | KV-cache == no-cache greedy |
| 07 | `07_phase7_tiny_gpt_shakespeare.md` | Tiny GPT | fluent Shakespeare-style text |
| 08 | `08_phase8_general_pretraining.md` | General pretrain | target perplexity; coherent text |
| 09 | `09_phase9_philosophy_dataset.md` | Philosophy corpus | dedup + metadata; multi-tradition |
| 10 | `10_phase10_domain_adaptation.md` | Domain adaptation | philosophy-val perplexity; stays fluent |
| 11 | `11_phaseR_reasoning_cot.md` | Reasoning / CoT | trace↔answer consistent; logic accuracy up |
| 12 | `12_phase11_instruction_dataset.md` | Instruction data | style linter passes; no fence-sitting |
| 13 | `13_phaseV_constitution_persona_sft.md` | Persona SFT | A/B prefers blunt+convicted voice |
| 14 | `14_phaseP_preference_dpo.md` | DPO | sycophancy down, conviction up, no regress |
| 15 | `15_phaseC_calibration_abstention.md` | Calibration | ECE under target; low unwarranted abstention |
| 16 | `16_phase12_evaluation_suite.md` | Evaluation suite | auto VERITAS report w/ deltas |
| 17 | `17_phase13_inference_api.md` | Inference API | endpoints work; persona server-side |
| 18 | `18_phase14_chat_app.md` | Chat app | end-to-end chat w/ reasoning+confidence UX |
| 19 | `19_phase15_experiment_tracking.md` | Experiment tracking | every run fully reproducible record |
| 20 | `20_phase16_testing.md` | Testing | 90%+ coverage; alignment code best-tested |
| 21 | `21_phase17_documentation.md` | Documentation | a new dev can reproduce a run |

## Source-of-truth docs (referenced by every phase)

- `Technical_Spec.md` — interfaces, shapes, hyperparameters, formats.
- `Detailed_Implementation_Plan.md` — the master plan and the VERITAS pillar.
- `Expected_Outcome.md` — the authoritative pass/fail gate for each phase.
- `Sample_Answers.md` / `Model_Answers_50.md` — the behavioral target (the voice).
- `configs/constitution.md`, `configs/persona_system_prompt.txt` — the character.

## Standard template (every phase file follows this)

`Goal` → `Prerequisites` → `Deliverables (files)` → `Tasks` → `Tests` → `Acceptance gate`
→ `Definition of done / commit` → `Cursor kickoff prompt`.
