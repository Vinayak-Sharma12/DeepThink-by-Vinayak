# Phase 17 — Documentation

## Goal
Documentation thorough enough that a new engineer can reproduce training and understand the
character design from the docs alone — including an honest account of the model's failure
modes and limitations (a truthfulness project documents its own weaknesses).

## Prerequisites
The model exists and has been evaluated (so limitations are real, not guessed).

## Deliverables (files)
```
README.md                 # exists — finalize quickstart
docs/architecture.md      # diagrams + module map
docs/training.md          # how to reproduce each stage
docs/inference.md         # API + generation usage
docs/developer.md         # repo conventions, adding modules, running tests
docs/veritas.md           # the character: constitution, stages, KNOWN FAILURE MODES
```

## Spec (from Technical_Spec.md §15, Detailed_Implementation_Plan.md §12)
- `docs/veritas.md` must include the character design, the constitution, the training stages,
  and — critically — the **known failure modes & limitations** (parametric knowledge can be
  stale/wrong; opinionated-by-design lean is a values config; no personal-cruelty guardrail;
  small-scale brittleness outside its domains).

## Tasks
1. Write architecture, training, inference, and developer guides.
2. Write `docs/veritas.md`: character, constitution, stages, and an honest limitations section.
3. Finalize the README quickstart (install → train tiny → generate → serve → chat).
4. Add architecture diagrams (the pipeline + the model block).

## Tests / checks
- A new engineer can follow the docs to reproduce a run and explain *why* the model behaves
  as it does.
- Limitations are stated honestly (omitting weaknesses is a fail for this project).

## Acceptance gate (from Expected_Outcome.md)
- Docs let someone reproduce a run and explain the character design; limitations honestly
  stated. (Red flag: docs describe architecture but omit the model's weaknesses.)

## Definition of done / commit
`docs: full documentation incl. docs/veritas.md with honest limitations`

## Cursor kickoff prompt
> Write the documentation per `phases/21_phase17_documentation.md`: architecture, training,
> inference, and developer guides; finalize the README quickstart; and write `docs/veritas.md`
> covering the character, constitution, training stages, and an honest **known failure modes &
> limitations** section. Add architecture diagrams. The limitations section is mandatory.
