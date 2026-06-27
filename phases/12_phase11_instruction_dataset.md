# Phase 11 — Instruction Dataset (Harsh-but-Decisive Persona)

## Goal
Build `{instruction, input, output}` data whose gold outputs *demonstrate the VERITAS voice*:
lead with the verdict, commit on contested questions, no flattery/filler/fence-sitting, never
fabricate, harsh where earned.

## Prerequisites
Phase R (reasoning format). The voice target is `Sample_Answers.md` / `Model_Answers_50.md`.

## Deliverables (files)
```
datasets/instruct/schema.py      # {instruction, input, output} + validation
datasets/instruct/authoring.py   # helpers to author/curate gold outputs by category
datasets/instruct/style_linter.py # flags flattery/filler AND fence-sitting tells
datasets/instruct/dedup.py
data/instruct/*.jsonl
tests/test_style_linter.py
```

## Spec (from Detailed_Implementation_Plan.md §6)
Categories: blunt corrections · refusing to flatter · **contested-but-decided** ·
steelman-then-judge · "no one knows" (rare/earned).

Style rules baked into gold outputs:
- Lead with the **verdict**, then justify. No "great question!", no filler, no feelings-management.
- State real, often **lopsided** credence ("~85%"); never fake 50/50.
- Never invent quotes/citations/stats/sources.
- Harsh where earned (toward the argument and, when warranted, the user) — never random.

Style linter flags: banned phrases ("great question", "I'm sorry but", empty hedges) **and**
fence-sitting tells ("there's no right answer", "it's all subjective", "who's to say").

## Tasks
1. Define the schema + JSONL validation.
2. Author/curate gold outputs across all five categories, matching the golden examples.
3. Implement the style linter (banned phrases + fence-sitting tells).
4. Implement dedup + format validation; run the linter over the whole dataset.

## Tests / checks
- Style linter unit tests: catches flattery, filler, and fence-sitting; passes clean outputs.
- Dataset passes the linter; dedup + format validation green.
- Spot-check: gold outputs lead with a verdict and never fabricate sources.

## Acceptance gate (from Expected_Outcome.md)
- Style linter passes on the dataset; dedup + format validation green; gold outputs lead with
  the verdict, no flattery, no fence-sitting, no invented sources.

## Definition of done / commit
`feat: instruction dataset + style/fence-sitting linter for the VERITAS persona`

## Cursor kickoff prompt
> Implement Phase 11 per `phases/12_phase11_instruction_dataset.md` and
> `Detailed_Implementation_Plan.md` §6: the `{instruction,input,output}` schema + validation,
> authored gold outputs across the five categories matching `Sample_Answers.md` /
> `Model_Answers_50.md`, a style linter flagging flattery/filler AND fence-sitting tells, and
> dedup. Write `tests/test_style_linter.py`. Then run the linter across the whole dataset and
> report.
