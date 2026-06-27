# Phase 13 — Inference API (FastAPI)

## Goal
Serve the model over HTTP with server-side persona injection, a reasoning-trace toggle, and a
confidence field.

## Prerequisites
Phase 6 (generation), a promoted checkpoint, `configs/persona_system_prompt.txt`.

## Deliverables (files)
```
app/api/main.py          # FastAPI app
app/api/schemas.py       # request/response models
app/api/model_runtime.py # load checkpoint + tokenizer; KV-cache generation
scripts/serve.py         # uvicorn entry (make serve)
tests/test_api.py
```

## Spec (from Technical_Spec.md §9)
| Endpoint | Method | Notes |
|---|---|---|
| `/generate` | POST | raw completion; `show_reasoning: bool` |
| `/chat` | POST | role-formatted; `show_reasoning: bool`; returns `confidence` |
| `/health` | GET | liveness |
| `/model` | GET | model + config metadata (`name: "DeepThink by Vinayak"`) |

- **Persona system prompt injected server-side by default** (must not depend on the client).
- `/chat` response includes `answer`, `confidence` (when emitted), `reasoning` (only if
  `show_reasoning=true`), `name`, `model`, `stage`.

## Tasks
1. Implement runtime: load checkpoint + tokenizer, KV-cache generation, think-span control.
2. Implement endpoints + Pydantic schemas; inject the persona prompt server-side.
3. Parse/return a `confidence` value when the model emits one.
4. Add `scripts/serve.py` (uvicorn).

## Tests / checks
- Endpoints return valid schemas; `/health` and `/model` work.
- `show_reasoning` toggles the think span in/out of the response.
- Persona is applied even when the client sends no system prompt.
- Latency acceptable with KV cache.

## Acceptance gate (from Expected_Outcome.md)
- API serves the model; reasoning toggle + confidence work end-to-end; persona applied by
  default; latency acceptable with KV cache.

## Definition of done / commit
`feat: FastAPI inference service with server-side persona, reasoning toggle, confidence`

## Cursor kickoff prompt
> Implement the inference API per `phases/17_phase13_inference_api.md` and `Technical_Spec.md`
> §9: a FastAPI app with `/generate`, `/chat`, `/health`, `/model`, KV-cache generation,
> server-side persona injection, a `show_reasoning` toggle, and a `confidence` field. Add
> `scripts/serve.py`. Write `tests/test_api.py` covering schemas, the reasoning toggle, and
> default persona application.
