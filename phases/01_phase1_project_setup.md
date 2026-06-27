# Phase 1 — Project Setup

## Goal
Create a clean, reproducible repository scaffold so every later phase has a home and the
environment reproduces with a single command.

## Prerequisites
None. This is the first phase.

## Deliverables (files)
```
pyproject.toml          # project metadata + deps + tool config (ruff/black/mypy/pytest)
requirements.txt        # pinned runtime deps (torch, numpy, fastapi, streamlit, etc.)
Makefile                # install / test / lint / format / train targets
.gitignore              # python, venv, models/, experiments/, data/, __pycache__
README.md               # already exists — link the phase docs
LICENSE                 # placeholder (TBD)
<package tree>          # empty packages with __init__.py + a smoke test
```

Package tree to create (each with `__init__.py`):
```
configs/ data/ datasets/ docs/
model/ tokenizer/ training/ evaluation/ inference/
experiments/ scripts/ tests/ models/ logs/ app/
```

## Tasks
1. Create the directory tree above; add `__init__.py` to every Python package.
2. `pyproject.toml`: set name, Python `>=3.10`, dependencies, and configure ruff + mypy +
   pytest. Use `dataclasses`, type hints, PEP8 (per `AGENTS.md`).
3. `requirements.txt`: pin `torch`, `numpy`, `pytest`, `pytest-cov`, `fastapi`, `uvicorn`,
   `streamlit`, `pyyaml`, `tqdm`. (Add only what's needed; justify extras.)
4. `Makefile` targets: `install`, `test`, `lint`, `format`, `coverage` (and stubs for
   `train`, `generate`, `serve`, `app`).
5. Add `tests/test_smoke.py` that imports every package and asserts it loads.
6. `git init`; first commit.

## Tests
- `tests/test_smoke.py`: imports `model`, `tokenizer`, `training`, `evaluation`,
  `inference` without error.

## Acceptance gate (from Expected_Outcome.md)
- Fresh clone + `make install` succeeds on a clean machine.
- `pytest` runs (even if mostly empty) and the smoke test passes.
- No undocumented manual setup steps ("works on my machine" is a fail).

## Definition of done / commit
`feat: project scaffold, tooling, and reproducible environment`

## Cursor kickoff prompt
> Create the repository scaffold described in `phases/01_phase1_project_setup.md`. Build the
> full package tree with `__init__.py` files, a `pyproject.toml` (Python ≥3.10, ruff + mypy +
> pytest configured), a pinned `requirements.txt`, a `Makefile` (`install`/`test`/`lint`/
> `format`/`coverage` + stubs for `train`/`generate`/`serve`/`app`), a `.gitignore`, and a
> `tests/test_smoke.py` that imports every package. Follow `AGENTS.md`. Do not add
> dependencies beyond those listed without justifying them.
