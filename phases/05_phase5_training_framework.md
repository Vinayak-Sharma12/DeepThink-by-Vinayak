# Phase 5 — Training Framework (CPU / MPS / CUDA)

## Goal
A from-scratch trainer with grad accumulation, mixed precision, cosine LR + warmup, grad
clipping, and bit-reproducible checkpoint/resume across CPU, Apple MPS, and CUDA.

## Prerequisites
Phase 4 (model), Phase 3 (batches).

## Deliverables (files)
```
training/optimizer.py    # AdamW config (decoupled weight decay)
training/scheduler.py    # cosine decay + linear warmup
training/checkpoint.py    # save/load full state
training/trainer.py       # training loop (accum, amp, clip, eval, logging)
training/device.py        # CPU/MPS/CUDA selection + dtype policy
tests/test_training.py
```

## Spec (from Technical_Spec.md §6)
| Feature | Spec |
|---|---|
| Grad accumulation | micro-batch → effective batch |
| Mixed precision | bf16 where supported |
| LR schedule | cosine decay + linear warmup |
| Grad clipping | global-norm (e.g. 1.0) |
| Grad checkpointing | enabled for Base tier |
| Seeding | deterministic; RNG saved/restored |

Checkpoint stores: `model_state, optimizer_state, scheduler_state, step, rng_state
(python/numpy/torch), GPTConfig, tokenizer ref, stage tag`.

## Tasks
1. Implement device/dtype selection (bf16 on CUDA/MPS where supported, fp32 fallback on CPU).
2. Implement AdamW config + cosine-with-warmup scheduler.
3. Implement the loop: forward → masked loss → backward with grad accumulation → clip → step
   → schedule; periodic eval + loss logging.
4. Implement checkpoint save/load capturing **all** state above.
5. Implement deterministic seeding and RNG save/restore.

## Tests
- Loss decreases on a tiny synthetic task.
- **Resume reproduces the loss curve**: train N steps, save, resume, and the next steps match
  a non-interrupted run (bit-reproducible).
- Grad accumulation of K micro-batches ≈ one batch of K× size (within tolerance).
- Grad clipping caps the global norm.

## Acceptance gate (from Expected_Outcome.md)
- Training loss decreases smoothly; resume-from-checkpoint matches the pre-save loss curve
  (bit-reproducible). No NaNs/spikes.

## Definition of done / commit
`feat: from-scratch trainer with amp, accumulation, cosine schedule, reproducible checkpoints`

## Cursor kickoff prompt
> Implement the training framework per `phases/05_phase5_training_framework.md` and
> `Technical_Spec.md` §6: device/dtype policy (CPU/MPS/CUDA, bf16), AdamW, cosine+warmup
> scheduler, the training loop (masked loss, grad accumulation, global-norm clipping, eval,
> logging), and full checkpoint save/load (model+opt+sched+step+RNG+config). Write
> `tests/test_training.py` proving loss decreases, grad accumulation equivalence, clipping,
> and **bit-reproducible resume**. No HuggingFace Trainer.
