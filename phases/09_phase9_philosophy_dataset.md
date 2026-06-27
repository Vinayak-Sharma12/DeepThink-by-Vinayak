# Phase 9 — Philosophy Dataset Builder (the LOGOS Corpus)

## Goal
Build a clean, deduplicated, richly-tagged JSONL corpus of philosophy / religion / ethics /
psychology / logic — the raw material that makes the model's eventual *verdicts* earned. This
is the project's long pole; do not rush it.

## Prerequisites
Phase 3 (cleaning utilities). Can be built in parallel with Phase 8 training.

## Deliverables (files)
```
datasets/corpus/ingest.py       # import books/texts (public-domain first)
datasets/corpus/clean.py        # clean + normalize
datasets/corpus/dedup.py        # near-duplicate removal
datasets/corpus/metadata.py     # metadata extraction + schema validation
datasets/corpus/split.py        # chapter/section splitting
datasets/corpus/build_jsonl.py  # emit final JSONL
data/logos/holdout/...          # CLEAN philosophy validation set (never trained on)
DATA_SOURCES.md                 # provenance + licensing per text
tests/test_corpus.py
```

## Spec (from Technical_Spec.md §5.3)
Each document carries:
```
title, author, year, school, religion, topic, language, stance, tradition
```
- `stance ∈ {argues_for, argues_against, survey, primary_source, commentary}`.
- **Balanced INPUT sampling** must be possible: the model must read every side (theist,
  atheist, each tradition) at its strongest. Balanced input ≠ balanced output.

## Tasks
1. Source public-domain / properly-licensed texts; record provenance in `DATA_SOURCES.md`.
2. Clean + normalize; split into chapters/sections.
3. Deduplicate (exact + near-duplicate); verify removal.
4. Extract/validate metadata against the schema; flag missing `stance`/`tradition`.
5. Emit JSONL; build a balanced-sampling helper keyed on `stance`/`tradition`.
6. Carve out a **clean held-out philosophy validation set** (never used in training).

## Tests / checks
- Dedup verified (no near-duplicates above threshold).
- Metadata coverage high; multiple traditions and both stances represented per major topic.
- Balanced sampler can draw multiple sides of a question.
- Held-out set is disjoint from training data.

## Acceptance gate (from Expected_Outcome.md)
- Corpus is queryable; dedup verified; metadata coverage high; multiple traditions
  represented; the balanced sampler can draw multiple sides. (Red flag: corpus skewed to one
  school → future "truth-seeking" is just bias.)

## Definition of done / commit
`feat: LOGOS philosophy corpus builder with stance/tradition metadata + holdout set`

## Cursor kickoff prompt
> Build the philosophy corpus tooling per `phases/09_phase9_philosophy_dataset.md`: ingest,
> clean, dedup, metadata extraction/validation (incl. `stance` and `tradition`), chapter
> splitting, JSONL emission, and a balanced-sampling helper. Record provenance in
> `DATA_SOURCES.md` and carve out a clean held-out philosophy validation set. Write
> `tests/test_corpus.py` for dedup, metadata coverage, balanced sampling, and holdout
> disjointness. Prioritize public-domain sources.
