"""Prepare general-corpus token shards at a fixed context length."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.sources.general import CorpusMixConfig, build_mix_config, prepare_general_shards
from scripts.train import (
    bootstrap_general_tokenizer_corpus,
    train_or_load_tokenizer,
)
from training.experiment import load_yaml_config

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Download, mix, and pack general-corpus shards.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "small.yaml",
        help="YAML config with data.mix and ctx_len settings",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild shards even if they already exist",
    )
    return parser


def prepare_from_config(config_path: Path, *, force: bool = False) -> dict[str, object]:
    """Prepare general shards and return summary metadata."""
    config = load_yaml_config(config_path)
    data_section = config.get("data")
    model_section = config.get("model")
    tokenizer_section = config.get("tokenizer")
    if not isinstance(data_section, dict):
        msg = "Config section 'data' must be a mapping"
        raise TypeError(msg)
    if not isinstance(model_section, dict):
        msg = "Config section 'model' must be a mapping"
        raise TypeError(msg)
    if not isinstance(tokenizer_section, dict):
        msg = "Config section 'tokenizer' must be a mapping"
        raise TypeError(msg)

    data_dir = ROOT / str(data_section.get("data_dir", "data/general"))
    ctx_len = int(data_section.get("ctx_len", model_section.get("ctx_len", 1024)))
    mix = build_mix_config(data_section)
    seed = int(data_section.get("seed", mix.seed))
    mix = CorpusMixConfig(**{**mix.__dict__, "seed": seed})

    tokenizer_dir = data_dir / "tokenizer"
    sample_max_chars = int(tokenizer_section.get("sample_max_chars", 8_000_000))
    corpus_text = bootstrap_general_tokenizer_corpus(
        data_dir,
        mix,
        max_chars=sample_max_chars,
    )
    tokenizer = train_or_load_tokenizer(
        config,
        corpus_text,
        tokenizer_dir,
        shared_dir=tokenizer_dir,
    )

    prepared = prepare_general_shards(
        data_dir,
        tokenizer,
        ctx_len,
        mix,
        force=force,
    )
    summary = {
        "data_dir": str(prepared.data_dir),
        "train_shard_dir": str(prepared.train_shard_dir),
        "val_shard_dir": str(prepared.val_shard_dir),
        "train_sequences": prepared.train_sequences,
        "val_sequences": prepared.val_sequences,
        "train_documents": prepared.train_documents,
        "val_documents": prepared.val_documents,
        "ctx_len": ctx_len,
    }
    meta_out = data_dir / "prepare_summary.json"
    meta_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Prepared %d train / %d val sequences at ctx_len=%d → %s",
        prepared.train_sequences,
        prepared.val_sequences,
        ctx_len,
        data_dir,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    summary = prepare_from_config(config_path, force=args.force)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
