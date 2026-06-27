# Phase C — Calibration & Earned Credence

## Goal
Make the model's confidence *mean* something so its strong verdicts are backed by real
credence — without sliding back into fence-sitting. Calibration here is honest credence
reporting, often **lopsided**, not a push toward 50%.

## Prerequisites
Phase P (DPO checkpoint).

## Deliverables (files)
```
evaluation/ece.py                  # Expected Calibration Error (from scratch)
evaluation/abstention.py           # abstention scorer
datasets/calibration/contested.py  # contested-but-decidable items (with ground truth)
datasets/calibration/unknowable.py # genuinely unanswerable items
training/calibration.py            # calibration tuning
experiments/experiment_00X/
tests/test_ece.py
tests/test_abstention.py
```

## Spec (from Technical_Spec.md §7.4)
- **Verbalized credence:** attach real confidence to claims, incl. lopsided (e.g. "~85%").
- **ECE** from scratch: bin stated confidence vs actual correctness. Target = *accuracy of
  the number*, NOT pushing numbers toward 50.
- **Reserved abstention:** "no one knows" only for the genuinely unknowable; using it as a
  dodge on a decidable question is penalized.

## Tasks
1. Implement ECE (confidence bins, per-bin accuracy, weighted gap) with tests.
2. Build the contested-but-decidable set (with ground truth) and the unknowable set.
3. Implement the abstention scorer: reward correct abstention on unknowables, penalize it on
   decidables, penalize confident fabrication heavily.
4. Run calibration tuning; re-measure ECE and abstention behavior.

## Tests / checks
- `test_ece.py`: ECE matches a hand-computed value on a toy distribution; perfect calibration → 0.
- `test_abstention.py`: scorer rewards/penalizes correctly across both sets.
- ECE below target threshold, **including on lopsided-credence items**.
- High fabrication-refusal on unknowables; **low** abstention on contested-but-decidable items.

## Acceptance gate (from Expected_Outcome.md)
- ECE below target (incl. lopsided items); high fabrication-refusal on true unknowables; low
  unwarranted-abstention on merely-contested questions.

## Definition of done / commit
`feat: calibration (from-scratch ECE) + reserved abstention; honest lopsided credence`

## Cursor kickoff prompt
> Implement Phase C per `phases/15_phaseC_calibration_abstention.md` and `Technical_Spec.md`
> §7.4: from-scratch ECE, an abstention scorer, a contested-but-decidable set (with ground
> truth) and an unknowable set, and calibration tuning. Write `tests/test_ece.py` and
> `tests/test_abstention.py`. Confirm ECE is low (incl. lopsided items), abstention is high on
> unknowables and low on decidables. Calibration must not collapse credences toward 50%.
