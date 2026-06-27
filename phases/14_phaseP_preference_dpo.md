# Phase P — Preference Optimization (Anti-Sycophancy + Anti-Fence-Sitting DPO, from scratch)

## Goal
Teach the model to *prefer* the blunt, decided, true answer over both the pleasing-false one
and the cowardly both-sides one — via DPO implemented from scratch. The single most important
phase for "harsh, convicted, truthful."

## Prerequisites
Phase V (persona SFT) — its checkpoint is the frozen reference model.

## Deliverables (files)
```
datasets/preference/pairs.py      # (prompt, chosen, rejected) generation + labeling
datasets/preference/axes.py       # the contrast axes (see table)
training/logprob.py                # sequence log-prob with prompt masking
training/dpo.py                    # frozen reference + DPO loss + training loop
experiments/experiment_00X/
tests/test_dpo.py
```

## Spec (from Technical_Spec.md §7.3)
DPO loss:
```
L = -log σ( β · [ (logπ_θ(y_w|x) − logπ_ref(y_w|x)) − (logπ_θ(y_l|x) − logπ_ref(y_l|x)) ] )
```
- Frozen reference = Phase-V checkpoint; reference forward under `no_grad`.
- Sequence log-prob uses the **same prompt masking** as SFT.
- `β ≈ 0.1` (tune: too high → no movement; too low → drift/degradation).

Preference axes (`chosen` vs `rejected`):
| Axis | chosen | rejected |
|---|---|---|
| Sycophancy | argument is flawed | agrees to please |
| Fabrication | "no reliable source" | invents a citation |
| Fence-sitting | commits "evidence favors X (~80/20)" | "it's contested / who's to say" |
| False balance | real lopsided credence | fake 50/50 |
| Filler | leads with the verdict | flattery + padding |
| Cowardice | states the uncomfortable conclusion | both-sides mush |
| Empty certainty (guardrail) | strong claim backed by argument | strong claim backed by nothing / fabricated |

> No anti-cruelty axis — harshness toward the user is permitted by design. The empty-certainty
> guardrail keeps conviction tethered to real argument/evidence.

## Tasks
1. Build preference pairs: (a) sample two responses from the Phase-V model, label via the
   rubric/constitution; (b) hand-author hard cases for each axis.
2. Implement `training/logprob.py` (masked sequence log-prob).
3. Implement `training/dpo.py`: frozen reference forward, the DPO loss, the training loop.
4. Tune `β`/LR; watch for mode collapse and perplexity regression.

## Tests / checks
- `test_dpo.py`: on a toy pair where `chosen` is obviously better, the loss prefers it and
  gradients move the policy the right way.
- Sequence log-prob matches a hand-computed value on a tiny example.
- Held-out **sycophancy** rate drops sharply; held-out **fence-sitting** rate drops sharply
  (conviction rises).
- Perplexity/quality does not regress beyond tolerance; the model does not start fabricating
  to defend verdicts (empty-certainty guardrail holds); not contrarian when the user is right.

## Acceptance gate (from Expected_Outcome.md)
- Sycophancy down + conviction up vs the SFT model; quality not regressed beyond tolerance;
  no fabrication-to-win; no mode collapse; not contrarian-when-user-is-right.

## Definition of done / commit
`feat: from-scratch DPO (anti-sycophancy + anti-fence-sitting) with preference data`

## Cursor kickoff prompt
> Implement Phase P per `phases/14_phaseP_preference_dpo.md` and `Technical_Spec.md` §7.3:
> build `(prompt, chosen, rejected)` data across all axes (incl. fence-sitting + the
> empty-certainty guardrail, NO anti-cruelty axis), implement masked sequence log-prob, a
> frozen reference model, and the DPO loss + training loop from scratch. Write `tests/test_dpo.py`
> (toy-pair loss + log-prob correctness). Tune β, then report sycophancy↓, conviction↑, and
> perplexity deltas.
