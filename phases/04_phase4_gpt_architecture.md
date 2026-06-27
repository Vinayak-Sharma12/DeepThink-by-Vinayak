# Phase 4 — GPT Architecture (RoPE / RMSNorm / SwiGLU / KV cache)

## Goal
Implement a decoder-only, pre-norm transformer from scratch as independent, tested modules,
all driven by a single `GPTConfig`. The phase passes when a Nano model overfits one batch.

## Prerequisites
Phase 3 (so you can feed it real batches), Phase 1 tooling.

## Deliverables (files)
```
model/config.py        # GPTConfig dataclass
model/embeddings.py     # token embedding (+ weight tying hook)
model/rope.py           # rotary position embeddings
model/rmsnorm.py        # RMSNorm
model/swiglu.py         # SwiGLU feed-forward
model/attention.py      # causal multi-head attention (+ GQA option, + KV cache)
model/block.py          # pre-norm transformer block
model/gpt.py            # full model, forward, count_parameters(), generate hook
tests/test_model.py
```

## Spec (from Technical_Spec.md §4)
```python
@dataclass
class GPTConfig:
    vocab_size: int = 32_000
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None      # None => MHA; < n_head => GQA
    d_model: int = 768
    ffn_mult: float = 8/3
    ctx_len: int = 2048
    rope_theta: float = 10_000.0
    rmsnorm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True
```
- Components: RoPE, RMSNorm (pre-norm), SwiGLU, causal MHA (GQA optional), KV cache, weight
  tying.
- Forward: `input_ids[B,T] -> logits[B,T,vocab]`; causal masking enforced.

## Tasks
1. Implement each module independently with its own unit test (shapes + a numerical sanity
   check). No module hardcodes dims — everything reads `GPTConfig`.
2. Implement RoPE and verify rotation is applied to q/k correctly (relative-position property).
3. Implement attention with an optional KV cache; support `n_kv_head` for GQA.
4. Assemble the block (pre-norm: `x + attn(norm(x))`, `x + ffn(norm(x))`) and the full model.
5. Implement weight tying (embedding ⇄ output projection) and `count_parameters()`.
6. Implement an `overfit_one_batch()` test path.

## Tests
- Per-module forward/backward shape tests (with and without KV cache).
- RoPE correctness (relative position invariance on a toy case).
- KV-cache forward matches no-cache forward for the same input (logits parity).
- **Overfit-one-batch: a Nano config drives loss to ~0** — the key gate.
- `count_parameters()` matches a hand-computed value for a tiny config.

## Acceptance gate (from Expected_Outcome.md)
- Nano model overfits a single batch to ~0 loss; each module has its own test; forward/
  backward shapes correct with and without KV cache. If loss won't reach ~0, **stop and fix**
  the wiring/masking bug before anything else.

## Definition of done / commit
`feat: from-scratch GPT (RoPE, RMSNorm, SwiGLU, GQA, KV cache, weight tying)`

## Cursor kickoff prompt
> Implement the decoder-only GPT per `phases/04_phase4_gpt_architecture.md` and
> `Technical_Spec.md` §4 as independent, unit-tested modules driven by `GPTConfig` (RoPE,
> RMSNorm, SwiGLU, causal MHA with optional GQA + KV cache, weight tying, `count_parameters()`).
> Write `tests/test_model.py` including per-module shape tests, RoPE correctness, KV-cache vs
> no-cache logits parity, and an overfit-one-batch test that drives a Nano config to ~0 loss.
> Do not copy an existing GPT implementation.
