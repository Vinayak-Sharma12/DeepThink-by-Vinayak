# Phase 12 — Evaluation (Truthfulness / VERITAS Suite)

## Goal
A from-scratch evaluation harness that scores every checkpoint on the VERITAS metrics and
auto-emits a `veritas_report.md` with deltas vs the previous best. No checkpoint is promoted
without it. **Build this early (around Phase R)** so every VERITAS iteration is judged.

## Prerequisites
Phase 6 (generation). Best built alongside Phase R; finalized before Phases V/P/C iterate.

## Deliverables (files)
```
evaluation/harness.py        # run all metrics over a checkpoint
evaluation/metrics/truthfulness.py
evaluation/metrics/sycophancy.py
evaluation/metrics/fabrication.py
evaluation/metrics/conviction.py        # verdict vs fence-sitting
evaluation/metrics/calibration.py       # uses evaluation/ece.py
evaluation/metrics/abstention.py
evaluation/metrics/contradiction.py
evaluation/metrics/bluntness.py         # uses the style linter
evaluation/metrics/reasoning.py
evaluation/report.py         # writes veritas_report.md with deltas
evaluation/datasets/...      # hand-built test sets per metric (in project domains)
tests/test_evaluation.py
```

## Spec (from Technical_Spec.md §10)
| Metric | Measures |
|---|---|
| Truthfulness | resists common false beliefs / leading questions |
| Sycophancy rate | caves when the user pushes a wrong claim |
| Fabrication rate | invents quotes/citations/facts |
| **Conviction rate** | reaches a verdict on contested Qs vs fence-sitting |
| Calibration (ECE) | accuracy of stated confidence, incl. lopsided |
| Unwarranted-abstention | "no one knows" on a decidable Q |
| Contradiction rate | verdict self-consistency across paraphrases |
| Bluntness/style | no flattery/filler; leads with the verdict |
| Reasoning quality | logic/fallacy/multi-step accuracy |

## Tasks
1. Build hand-crafted test sets in the project's domains (TruthfulQA-style; paired
   neutral/leading prompts for sycophancy; source-demands for fabrication; contested sets for
   conviction; paraphrase sets for contradiction).
2. Implement each metric scorer (reuse `ece.py`, abstention scorer, style linter).
3. Implement `report.py` to write `veritas_report.md` with deltas vs previous best.
4. Wire the harness to run automatically after relevant training runs.

## Tests / checks
- Each scorer unit-tested on tiny fixtures with known answers.
- Report generation produces a complete table with deltas.
- The promotion rule is enforced programmatically.

## Acceptance gate (from Expected_Outcome.md)
- Every experiment auto-emits a report with deltas; **no checkpoint promoted** unless
  truthfulness/sycophancy/fabrication/conviction improve or hold while helpfulness doesn't
  regress. (Red flag: cherry-picked samples instead of the full metric table.)

## Definition of done / commit
`feat: VERITAS evaluation suite + auto veritas_report.md with promotion gate`

## Cursor kickoff prompt
> Implement the VERITAS evaluation suite per `phases/16_phase12_evaluation_suite.md` and
> `Technical_Spec.md` §10: hand-built domain test sets, from-scratch scorers for every metric
> (truthfulness, sycophancy, fabrication, conviction, ECE, abstention, contradiction,
> bluntness, reasoning), and `report.py` that writes `veritas_report.md` with deltas and
> enforces the promotion gate. Write `tests/test_evaluation.py` for each scorer on fixtures.
