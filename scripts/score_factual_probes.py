"""Score corpus-grounded factual probes against a trained checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.sources.general import build_mix_config, iter_source_documents
from evaluation.factual_probes import (
    load_probes,
    score_factual_probes,
    verify_in_corpus,
    write_probe_results,
)
from inference.loader import load_gpt_from_checkpoint
from inference.sampler import SamplerConfig
from tokenizer.tokenizer import Tokenizer
from training.device import resolve_device
from training.experiment import load_yaml_config

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Score factual probes against a LOGOS checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--probes",
        type=Path,
        default=ROOT / "evaluation" / "data" / "factual_probes.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "small.yaml")
    parser.add_argument("--verify-corpus", action="store_true", default=True)
    parser.add_argument("--no-verify-corpus", action="store_false", dest="verify_corpus")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load a checkpoint and score factual probes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    probes = load_probes(args.probes)
    if args.verify_corpus:
        config = load_yaml_config(args.config if args.config.is_absolute() else ROOT / args.config)
        data_section = config.get("data", {})
        if not isinstance(data_section, dict):
            data_section = {}
        data_dir = ROOT / str(data_section.get("data_dir", "data/general"))
        mix = build_mix_config(data_section)
        documents = list(iter_source_documents(data_dir, mix))
        verified, dropped = verify_in_corpus(probes, documents)
        logger.info(
            "Corpus verification — kept %d probes, dropped %d",
            len(verified),
            len(dropped),
        )
        probes = verified

    tokenizer = Tokenizer.load(str(args.tokenizer))
    model = load_gpt_from_checkpoint(args.checkpoint, device)
    sampler = SamplerConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    results = score_factual_probes(
        model,
        tokenizer,
        probes,
        sampler=sampler,
        device=device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.parent / "factual_probes.json"
    write_probe_results(output_path, results)

    print(
        f"Factual probe hit-rate: {results['hits']}/{results['total']} "
        f"({results['hit_rate']:.1%})"
    )
    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
