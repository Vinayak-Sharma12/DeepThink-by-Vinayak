# Phase 16 — Testing (90%+ coverage)

## Goal
Reach 90%+ coverage with the **alignment-critical** code as the best-tested in the repo.
Coverage of trivial code with weak coverage of DPO/calibration/masking is a fail.

## Prerequisites
All implementation phases (tests are written alongside each, consolidated here).

## Deliverables (files)
```
tests/  (consolidated suite + coverage config)
Makefile: coverage target
```

## Spec (from Technical_Spec.md §12)
Highest-priority (load-bearing) tests:
- loss masking (prompt zeroed, completion counted; no off-by-one)
- sequence log-prob (used by DPO)
- DPO loss on a toy pair where `chosen` is obviously better
- ECE computation
- style linter (banned phrases + fence-sitting tells)
- abstention scorer
- overfit-one-batch (Nano → ~0 loss)
- KV-cache vs no-cache greedy parity

## Tasks
1. Audit coverage; fill gaps, prioritizing the load-bearing list above.
2. Add integration tests for the full pipeline on a tiny config (data → train → generate).
3. Add a coverage gate to CI/`make coverage` (fail under 90%).
4. Ensure deterministic tests (seeded; no flakiness).

## Tests / checks
- `pytest` green; coverage ≥ 90%.
- The alignment-critical modules have the highest coverage in the repo.

## Acceptance gate (from Expected_Outcome.md)
- `pytest` green; coverage ≥ 90%; the load-bearing VERITAS pieces are the best-tested code.

## Definition of done / commit
`test: reach 90%+ coverage with alignment-critical code best-tested`

## Cursor kickoff prompt
> Consolidate and complete the test suite per `phases/20_phase16_testing.md` and
> `Technical_Spec.md` §12 to reach 90%+ coverage, prioritizing the load-bearing list (loss
> masking, sequence log-prob, DPO loss, ECE, style linter, abstention, overfit-one-batch,
> KV-cache parity). Add tiny-config integration tests and a coverage gate to `make coverage`.
> Keep tests deterministic.
