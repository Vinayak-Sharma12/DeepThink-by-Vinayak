# Project LOGOS

> **Model:** DeepThink by Vinayak (internal codename: LOGOS · character pillar: VERITAS)

A GPT-style language model built **entirely from scratch in PyTorch**, specialized in
**philosophy, religion, spirituality, psychology, ethics, and logic** — and engineered to
have a specific character, **VERITAS**: truthful, blunt, opinionated, and decisive.

> This is not a neutral both-sides referee. LOGOS is designed to behave like someone who has
> read every side — every theist, every atheist, every tradition — and *made up their mind*.
> It commits to a verdict, states its real (often lopsided) confidence, refuses to fabricate,
> and is willing to be harsh. By design it has **no personal-cruelty guardrail**; its only
> anchor is that harshness must be earned by the quality of the argument.

## Status

Pre-implementation. The planning and specification are complete; coding starts at Phase 1.

## Documents

| Doc | What it is |
|---|---|
| `OverallPlan.md` | The original 17-phase plan (the backbone) |
| `Detailed_Implementation_Plan.md` | Master engineering plan incl. the VERITAS pillar |
| `Technical_Spec.md` | Exact technical surface: interfaces, shapes, hyperparameters, formats |
| `Expected_Outcome.md` | Per-phase artifacts, observable outcomes, and pass/fail gates |
| `Time_Estimates.md` | Realistic schedule per phase |
| `Sample_Answers.md` / `Model_Answers_50.md` | Golden behavioral target (the model's voice) |
| `configs/constitution.md` | The 10 principles that define the character |
| `configs/persona_system_prompt.txt` | Production system prompt encoding the constitution |

## Approach

Built from scratch — custom BPE tokenizer, custom decoder-only GPT (RoPE / RMSNorm / SwiGLU /
KV cache), custom trainer, and a from-scratch alignment stack (reasoning SFT, persona SFT,
DPO, calibration). No HuggingFace `Trainer`, no copied implementations.

### Model size ladder

| Stage | Params | Purpose |
|---|---|---|
| Nano | ~2M | wiring / overfit-a-batch sanity |
| Tiny | ~10M | Shakespeare |
| Small | 30–50M | general pretrain |
| Medium | ~70–90M | intermediate pretrain/domain bridge |
| Base | 110–160M | domain + reasoning + persona (carries VERITAS) |

### Training pipeline (order matters)

```
pretrain → domain-adapt → reasoning SFT → persona SFT → DPO → calibration → evaluate → promote
```

## Getting started

> Phase 1 deliverables (repo scaffold, `pyproject.toml`, `requirements.txt`, `Makefile`,
> `.gitignore`) are created in the first milestone. Once present:

```bash
make install      # set up the environment
pytest            # run the test suite
```

## License

TBD.
