# Phase R — Reasoning & Chain-of-Thought

## Goal
Teach the model to *deliberate, then decide*: produce a useful hidden `<|think|>` trace that
weighs the sides and ends in a verdict, followed by a final answer consistent with it.

## Prerequisites
Phase 10 (domain-adapted Base), Phase 3 loss masking, Phase 6 think-span decoding.

## Deliverables (files)
```
datasets/reasoning/dialectic.py   # synthetic thesis/antithesis/synthesis trace generator
datasets/reasoning/fallacy.py     # premise -> fallacy-name -> explanation corpus
datasets/reasoning/worked.py      # multi-step logic/math problems with full traces
datasets/reasoning/cite.py        # claim + <|cite|>justification<|/cite|> rewrites
datasets/reasoning/format.py      # assemble into the <|think|> training format
training/sft.py                   # masked-loss SFT loop (reused by V too)
experiments/experiment_00X/
tests/test_reasoning_format.py
```

## Spec (from Technical_Spec.md §7.1)
Training format:
```
<|system|> <persona/constitution>
<|user|>   <question>
<|assistant|>
<|think|>
  - restate the question; known vs assumed
  - steelman each candidate answer at its strongest
  - strongest counter-argument
  - weigh evidence; decide which side wins and by how much
<|/think|>
<final answer: leads with the verdict, blunt, decided>
```
- Train the `<|think|>` span; suppressible at inference.
- SFT next-token loss with **prompt masked**, loss only on the assistant span.
- The deliberation must end in a **verdict**, not a shrug.

## Tasks
1. Build the four reasoning data generators; emit examples in the format above.
2. Implement `training/sft.py` (reuses the Phase 5 trainer + Phase 3 masking).
3. Continue-train the Base model on reasoning data.
4. Hold out a reasoning eval set (logic/fallacy + dialectic).

## Tests / checks
- Format validator: every example has a well-formed think span and a final verdict.
- Trace↔answer consistency on the held-out set (answer follows from the trace).
- Logic/fallacy accuracy beats the pre-R checkpoint by a clear margin.
- The trace ends in a decision, not "both sides have a point."

## Acceptance gate (from Expected_Outcome.md)
- Coherent `<|think|>` trace; final answer consistent with it; logic/fallacy accuracy beats
  pre-R by a clear margin; the trace ends in a verdict.

## Definition of done / commit
`feat: reasoning/CoT data + reasoning SFT (deliberate then decide)`

## Cursor kickoff prompt
> Implement Phase R per `phases/11_phaseR_reasoning_cot.md` and `Technical_Spec.md` §7.1:
> build the dialectic, fallacy, worked-solution, and citation-rewrite data generators in the
> `<|think|>`→verdict format; implement `training/sft.py` (masked-loss, reusing the Phase 5
> trainer); continue-train the Base model; and create a held-out reasoning eval. Write
> `tests/test_reasoning_format.py` validating the format and trace→verdict structure. Verdicts,
> not shrugs.
