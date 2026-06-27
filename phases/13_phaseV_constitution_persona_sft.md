# Phase V — Constitution + Persona SFT

## Goal
Make the VERITAS character the model's *default* via the written constitution + a system
prompt + supervised fine-tuning, so the persona holds even when the system prompt is omitted.

## Prerequisites
Phase R (reasoning), Phase 11 (instruction data). Constitution + prompt already drafted in
`configs/`.

## Deliverables (files)
```
configs/constitution.md           # exists — finalize the 10 principles
configs/persona_system_prompt.txt  # exists — finalize second-person prompt
datasets/persona/dialogues.py      # multi-turn constitution-demonstration dialogues
scripts/train.py                   # reused for persona SFT
experiments/experiment_00X/
tests/test_persona_format.py
```

## Spec (from Technical_Spec.md §7.2, Detailed_Implementation_Plan.md §7)
- Prepend the persona system prompt in the `<|system|>` slot for **all** SFT/preference data
  and at inference (training with it present makes the persona robust).
- Fine-tune on Phase 11 instruction data + Phase R reasoning data + constitution-demonstration
  dialogues (the model visibly obeys each principle, **including committing to verdicts**).
- Loss masked to assistant turns only.

## Tasks
1. Finalize `constitution.md` (10 principles) and `persona_system_prompt.txt`.
2. Author multi-turn dialogues that each demonstrate constitution principles (esp. #2 commit,
   #4 harsh-but-earned, #5 steelman, #6 no fabrication).
3. Run persona SFT with the system prompt present and assistant-only loss.
4. Build an A/B comparison harness (pre-V vs post-V on the same prompts).

## Tests / checks
- Persona-format validator (system slot present, assistant-only loss spans).
- Blind A/B: raters/rubric prefer post-V for bluntness + conviction + honesty, no helpfulness drop.
- Post-V reaches verdicts on contested prompts more often than pre-V.
- Persona persists even when the system prompt is omitted (internalized, not prompt-dependent).

## Acceptance gate (from Expected_Outcome.md)
- Default voice is blunt, opinionated, decisive; obeys the 10 principles even without the
  system prompt; A/B prefers post-V for bluntness+conviction+honesty with no helpfulness drop.

## Definition of done / commit
`feat: constitution + persona system prompt + persona SFT (default VERITAS voice)`

## Cursor kickoff prompt
> Implement Phase V per `phases/13_phaseV_constitution_persona_sft.md`,
> `Detailed_Implementation_Plan.md` §7, and `Technical_Spec.md` §7.2. Finalize the constitution
> and persona prompt in `configs/`, author constitution-demonstration dialogues, run persona
> SFT (system prompt present, assistant-only loss) on the instruction + reasoning + dialogue
> data, and build an A/B harness comparing pre-V vs post-V. Confirm the persona persists with
> the system prompt removed.
