# Phase 14 — Chat Application (Streamlit)

## Goal
A usable chat UI that shows how the model reasoned and how confident it is — honest UX that
matches the model's own values.

## Prerequisites
Phase 13 (API).

## Deliverables (files)
```
app/chat/streamlit_app.py   # the UI
app/chat/client.py          # calls the FastAPI service
scripts/app.py              # launch (make app)
```

## Spec (from Technical_Spec.md §9 / Detailed_Implementation_Plan.md §11)
- Chat with conversation history and model selection.
- **"Show the model's reasoning"** toggle (reveals the `<|think|>` trace).
- **Confidence display** for emitted confidences.
- Parameter controls (temperature, top-k, top-p, max tokens).
- An honest banner: *"This model is opinionated and blunt by design. It takes sides, it can
  be harsh, and it can be wrong — check its reasoning and its sources, and argue back."*

## Tasks
1. Build the chat UI (history, input, send).
2. Wire it to the API client; stream or display responses.
3. Add the reasoning toggle, confidence display, parameter controls, and the banner.
4. Add model selection (which checkpoint/stage).

## Tests / checks (mostly manual + light)
- End-to-end conversation works against the running API.
- Reasoning toggle reveals/hides the trace; confidence shows when present.
- The banner is visible (honest UX — does not hide uncertainty/reasoning).

## Acceptance gate (from Expected_Outcome.md)
- A usable conversation; user can see how the model reasoned and how confident it is; UX
  matches the model's stated values (no hiding uncertainty/reasoning).

## Definition of done / commit
`feat: Streamlit chat app with reasoning toggle, confidence display, honesty banner`

## Cursor kickoff prompt
> Implement the Streamlit chat app per `phases/18_phase14_chat_app.md`: chat with history and
> model selection, a "Show the model's reasoning" toggle, a confidence display, parameter
> controls, and the honesty banner. Wire it to the FastAPI service via a small client. Add
> `scripts/app.py`. Confirm an end-to-end conversation works.
