"""Minimal Streamlit UI for testing a trained LOGOS checkpoint."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from inference.generate import GenerationConfig, generate
from inference.loader import load_gpt_from_checkpoint
from inference.sampler import SamplerConfig, make_generator
from model.gpt import GPT
from tokenizer import Tokenizer
from training.device import resolve_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "experiments" / "experiment_004" / "tiny_shakespeare_best.pt"
DEFAULT_TOKENIZER = ROOT / "experiments" / "experiment_004" / "tokenizer"

PRESET_PROMPTS = {
    "ROMEO (blank)": "ROMEO:\n",
    "First Citizen": "First Citizen:\nBefore we proceed any further, hear me speak.",
    "King Henry VI": "KING HENRY VI:\n",
    "To be or not": "To be or not to be,",
}


@st.cache_resource(show_spinner="Loading model…")
def load_resources(
    checkpoint_path: str,
    tokenizer_path: str,
    device_name: str | None,
) -> tuple[GPT, Tokenizer, torch.device]:
    """Load model and tokenizer once per checkpoint/device selection."""
    device = resolve_device(device_name)
    tokenizer = Tokenizer.load(tokenizer_path)
    model = load_gpt_from_checkpoint(Path(checkpoint_path), device)
    return model, tokenizer, device


def main() -> None:
    """Run the Streamlit playground."""
    st.set_page_config(page_title="LOGOS Playground", page_icon="🎭", layout="wide")
    st.title("LOGOS — Tiny Shakespeare playground")
    st.caption(
        "Completion model (not chat). Give a Shakespeare-style prompt; the model continues the text."
    )

    with st.sidebar:
        st.header("Checkpoint")
        checkpoint = st.text_input(
            "Weights (.pt)",
            value=str(DEFAULT_CHECKPOINT),
        )
        tokenizer_dir = st.text_input(
            "Tokenizer directory",
            value=str(DEFAULT_TOKENIZER),
        )
        device_choice = st.selectbox("Device", options=("auto", "cpu", "mps", "cuda"))
        device_name = None if device_choice == "auto" else device_choice

        st.header("Sampling")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
        max_new_tokens = st.slider("Max new tokens", min_value=16, max_value=512, value=120, step=8)
        seed = st.number_input("Seed (0 = random)", min_value=0, value=42, step=1)
        top_k = st.number_input("Top-k (0 = off)", min_value=0, value=0, step=1)
        top_p = st.slider("Top-p (1 = off)", min_value=0.05, max_value=1.0, value=1.0, step=0.05)

        preset = st.selectbox("Preset prompt", options=["(custom)"] + list(PRESET_PROMPTS.keys()))

    prompt_default = PRESET_PROMPTS.get(preset, "ROMEO:\n") if preset != "(custom)" else "ROMEO:\n"
    prompt = st.text_area("Prompt", value=prompt_default, height=120)

    col_generate, col_clear = st.columns([1, 5])
    with col_generate:
        run = st.button("Generate", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear output"):
            st.session_state.pop("last_output", None)
            st.rerun()

    if not Path(checkpoint).is_file():
        st.error(f"Checkpoint not found: {checkpoint}")
        return
    if not Path(tokenizer_dir).is_dir():
        st.error(f"Tokenizer directory not found: {tokenizer_dir}")
        return

    if run:
        try:
            model, tokenizer, device = load_resources(checkpoint, tokenizer_dir, device_name)
        except Exception as exc:
            st.exception(exc)
            return

        generator = make_generator(device, int(seed))

        sampler = SamplerConfig(
            temperature=temperature,
            top_k=int(top_k) if top_k > 0 else None,
            top_p=float(top_p) if top_p < 1.0 else None,
        )
        config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            sampler=sampler,
            add_bos=False,
        )

        with st.spinner("Generating…"):
            result = generate(model, tokenizer, prompt, config, device=device, generator=generator)

        completion_only = tokenizer.decode(result.token_ids, skip_special=True)
        st.session_state["last_output"] = {
            "full": result.text,
            "completion": completion_only,
            "tokens": len(result.token_ids),
        }

    if "last_output" in st.session_state:
        out = st.session_state["last_output"]
        st.subheader("Full output (prompt + completion)")
        st.text_area("full", value=out["full"], height=220, label_visibility="collapsed")
        st.subheader("Completion only")
        st.text_area("completion", value=out["completion"], height=160, label_visibility="collapsed")
        st.caption(f"Generated {out['tokens']} tokens")


if __name__ == "__main__":
    main()
