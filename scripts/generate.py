"""CLI entry point for text generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from inference.generate import GenerationConfig
from inference.generate import generate as run_generate
from inference.sampler import SamplerConfig
from model import GPT
from tokenizer import Tokenizer
from training.checkpoint import gpt_config_from_dict, load_checkpoint
from training.device import resolve_device
from training.optimizer import AdamWConfig, create_adamw
from training.scheduler import CosineWarmupConfig, create_cosine_warmup_scheduler


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the generate CLI."""
    parser = argparse.ArgumentParser(description="Generate text with a LOGOS checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer dir")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--thinking",
        choices=("show", "hide"),
        default="show",
        help="Show or hide <|think|>...<|/think|> span in output",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> GPT:
    """Load a ``GPT`` model from a training checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = gpt_config_from_dict(payload["gpt_config"])
    model = GPT(config)
    optimizer = create_adamw(model.parameters(), AdamWConfig())
    scheduler = create_cosine_warmup_scheduler(
        optimizer,
        CosineWarmupConfig(warmup_steps=1, max_steps=2),
    )
    load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    """Run generation and print the result to stdout."""
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    model = load_model_from_checkpoint(args.checkpoint, device)

    sampler = SamplerConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        sampler=sampler,
        thinking_mode=args.thinking,
    )
    result = run_generate(model, tokenizer, args.prompt, config, device=device)
    print(result.text)
    if args.thinking == "hide" and result.thinking is not None:
        print("\n--- reasoning (hidden) ---", file=sys.stderr)
        print(result.thinking.strip(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
