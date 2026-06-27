# CURSOR_IMPLEMENTATION_PLAN.md

# Project LOGOS — Cursor Implementation Plan

## Objective

Build a GPT-style language model completely from scratch using PyTorch, with every major component implemented within this repository. The final model should be specialized in philosophy, religion, spirituality, psychology, ethics, and logic.

Cursor should be used as an implementation partner while maintaining a clean, modular, production-quality codebase.

---

# Cursor Working Rules

## General Rules

* Never use HuggingFace Trainer.
* Never copy an existing GPT implementation.
* Build every major component from scratch.
* Keep every module independent.
* Write clean, documented Python code.
* Use type hints everywhere.
* Use dataclasses where appropriate.
* Follow PEP8.
* Every public function should contain docstrings.
* Every module should include unit tests.
* Never leave TODOs unless explicitly requested.

---

# Development Workflow

Every feature follows this cycle:

1. Design
2. Implement
3. Unit Test
4. Integration Test
5. Benchmark
6. Documentation
7. Git Commit

Never skip any step.

---

# Phase 1 — Project Setup

## Cursor Task

Create the entire repository structure.

```
logos/

configs/
data/
datasets/
docs/

model/
tokenizer/
training/
evaluation/
inference/

experiments/

scripts/

tests/

models/

logs/

app/
```

### Deliverables

* pyproject.toml
* requirements.txt
* Makefile
* .gitignore
* README.md

---

# Phase 2 — Tokenizer

## Cursor Task

Implement Byte Pair Encoding completely from scratch.

Modules

```
tokenizer/

dataset.py
vocab.py
trainer.py
encoder.py
decoder.py
serialization.py
```

Features

* Train tokenizer
* Save vocabulary
* Load vocabulary
* Encode
* Decode
* Unknown token handling

Tests

* Encoding correctness
* Decoding correctness
* Vocabulary generation
* Merge operations

---

# Phase 3 — Dataset Pipeline

Cursor should implement

* Text cleaning
* Chunk generation
* Dataset class
* Sequence generation
* Batch collation
* Random sampling

Support

* Tiny Shakespeare
* TinyStories
* WikiText

---

# Phase 4 — GPT Architecture

Cursor should implement every module separately.

Modules

```
Embedding

Position Embedding

LayerNorm

Feed Forward

Self Attention

Multi Head Attention

Transformer Block

GPT Model
```

Each module must have

* implementation
* documentation
* tests

---

# Phase 5 — Training Framework

Implement

* Trainer
* Optimizer
* Scheduler
* Gradient clipping
* Checkpoint saving
* Resume training
* Validation
* Loss logging

Training loop must support

* CPU
* Apple MPS
* CUDA

---

# Phase 6 — Text Generation

Implement

Sampling strategies

* Greedy
* Temperature
* Top-K
* Top-P

CLI

```
python generate.py
```

---

# Phase 7 — Tiny GPT Training

Dataset

Tiny Shakespeare

Target

10M parameters

Expected Result

Model generates Shakespeare-style text.

---

# Phase 8 — General Language Pretraining

Datasets

* TinyStories
* WikiText
* OpenWebText (subset)

Target

20M–50M parameters

---

# Phase 9 — Philosophy Dataset Builder

Cursor should create tools to

* Import books
* Clean text
* Remove duplicates
* Extract metadata
* Split chapters
* Generate JSONL

Metadata

```
title

author

year

school

religion

topic

language
```

---

# Phase 10 — Domain Adaptation

Continue training using

LOGOS Corpus

Implement

* Resume checkpoints
* Curriculum learning
* Validation
* Evaluation

---

# Phase 11 — Instruction Dataset

Create tools for

```
instruction

input

output
```

Support JSONL.

Implement

* validation
* deduplication
* formatting

---

# Phase 12 — Evaluation

Metrics

* Perplexity
* Token Accuracy
* Loss
* Hallucination Rate
* Consistency
* Contradiction Detection

Generate reports after every experiment.

---

# Phase 13 — Inference API

Implement

FastAPI

Endpoints

```
POST /generate

POST /chat

GET /health

GET /model
```

---

# Phase 14 — Chat Application

Implement

Streamlit UI

Features

* Chat
* Parameters
* Conversation history
* Model selection
* Temperature control

---

# Phase 15 — Experiment Tracking

Each experiment should automatically create

```
experiments/

experiment_001/

config.yaml

metrics.json

loss.csv

generation_samples.md
```

---

# Phase 16 — Testing

Target

90%+ test coverage

Test

* tokenizer
* attention
* transformer
* trainer
* inference
* generation

---

# Phase 17 — Documentation

Generate

README.md

Architecture diagrams

API documentation

Training guide

Inference guide

Developer guide

---

# Git Strategy

Commit after every completed milestone.

Example

```
feat: implement tokenizer

feat: implement attention

feat: add transformer block

feat: add trainer

feat: support MPS

feat: philosophy dataset builder
```

---

# Cursor Prompting Strategy

Never ask Cursor to build the entire project in one prompt.

Instead, work milestone by milestone.

Example prompts:

* "Implement a Byte Pair Encoding tokenizer from scratch in Python with unit tests."

* "Implement a GPT-style Multi-Head Self-Attention module in PyTorch with causal masking and documentation."

* "Implement a production-ready trainer supporting CPU, CUDA, and Apple MPS."

* "Review this implementation for correctness, performance, and code quality without changing the architecture."

Treat each milestone as a pull request. Complete it, test it, commit it, then move to the next.

---

# Success Criteria

The project is complete when it includes:

* Custom tokenizer
* Custom GPT implementation
* Custom training pipeline
* Domain-specific philosophy corpus
* Domain-adapted language model
* Evaluation framework
* Chat application
* Complete documentation
* Fully reproducible training pipeline

The final repository should demonstrate a complete understanding of the architecture, training, and deployment of a GPT-style language model specialized in philosophy, religion, spirituality, ethics, psychology, and logic.
