# Phase 15 — Experiment Tracking

## Goal
Every training run auto-creates a complete, reproducible record — including which VERITAS
stages produced the checkpoint — so the model's character lineage is fully auditable. Build
this **early** (it pays off across all VERITAS iterations).

## Prerequisites
Phase 5 (trainer). Ideally implemented before Phases 8/R so every run is recorded.

## Deliverables (files)
```
experiments/tracker.py     # create + populate run folders
experiments/registry.py    # index runs; find previous best; lineage tags
tests/test_tracking.py
```

## Spec (from Technical_Spec.md §11)
Each run auto-creates `experiments/experiment_XXX/`:
```
config.yaml          # full GPTConfig + training config
metrics.json         # all eval metrics
loss.csv             # loss curve
generation_samples.md
veritas_report.md    # metric table + deltas, tagged with R/V/P/C lineage
```

## Tasks
1. Implement run-folder creation with an auto-incrementing id.
2. Persist `config.yaml`, `loss.csv`, `generation_samples.md` during/after training.
3. Record the **stage lineage** (which of R/V/P/C produced this checkpoint).
4. Implement a registry to find the previous best for delta computation.
5. Hook the tracker into the trainer + eval harness.

## Tests / checks
- A run produces all required files.
- Lineage tags are recorded and queryable.
- The registry correctly identifies the previous best for deltas.

## Acceptance gate (from Expected_Outcome.md)
- Every training run produces a complete, reproducible record; any model's character lineage
  is fully auditable. (Red flag: results exist but the config/lineage is lost.)

## Definition of done / commit
`feat: experiment tracking with auto run records + character lineage`

## Cursor kickoff prompt
> Implement experiment tracking per `phases/19_phase15_experiment_tracking.md` and
> `Technical_Spec.md` §11: auto-created `experiments/experiment_XXX/` folders with
> `config.yaml`, `metrics.json`, `loss.csv`, `generation_samples.md`, and `veritas_report.md`,
> plus stage-lineage tags and a registry that finds the previous best. Hook it into the trainer
> and eval harness. Write `tests/test_tracking.py`.
