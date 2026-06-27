"""Generic YAML-driven GPT pretraining entry point."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.collate import make_collate_fn
from datasets.dataset import PretrainDataset
from datasets.shakespeare import load_split_texts, prepare_shakespeare_splits
from inference.generate import GenerationConfig, generate
from inference.sampler import SamplerConfig
from model.config import GPTConfig
from model.gpt import GPT, count_parameters, estimate_parameter_count
from tokenizer.tokenizer import Tokenizer
from training.experiment import (
    LossRecord,
    copy_config,
    load_yaml_config,
    next_experiment_dir,
    write_loss_csv,
    write_metrics,
    write_samples,
)
from training.optimizer import AdamWConfig
from training.scheduler import CosineWarmupConfig
from training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train a LOGOS GPT from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "tiny.yaml",
        help="Path to training YAML config",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from",
    )
    return parser


def _require_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required mapping section from ``config``."""
    section = config.get(key)
    if not isinstance(section, dict):
        msg = f"Config section {key!r} must be a mapping"
        raise TypeError(msg)
    return section


def build_gpt_config(model_section: dict[str, Any], vocab_size: int) -> GPTConfig:
    """Build ``GPTConfig`` from YAML model settings."""
    return GPTConfig(
        vocab_size=vocab_size,
        n_layer=int(model_section.get("n_layer", 6)),
        n_head=int(model_section.get("n_head", 6)),
        n_kv_head=model_section.get("n_kv_head"),
        d_model=int(model_section.get("d_model", 384)),
        ffn_mult=float(model_section.get("ffn_mult", 8 / 3)),
        ctx_len=int(model_section.get("ctx_len", 512)),
        rope_theta=float(model_section.get("rope_theta", 10_000.0)),
        rmsnorm_eps=float(model_section.get("rmsnorm_eps", 1e-5)),
        dropout=float(model_section.get("dropout", 0.0)),
        tie_weights=bool(model_section.get("tie_weights", True)),
    )


def train_or_load_tokenizer(
    config: dict[str, Any],
    train_text: str,
    tokenizer_dir: Path,
    *,
    shared_dir: Path | None = None,
) -> Tokenizer:
    """Train a tokenizer on ``train_text`` or load an existing one."""
    tokenizer_section = _require_section(config, "tokenizer")
    vocab_size = int(tokenizer_section["vocab_size"])
    should_train = bool(tokenizer_section.get("train", True))
    existing_path = tokenizer_section.get("path")

    if existing_path:
        return Tokenizer.load(str(existing_path))

    for candidate in (shared_dir, tokenizer_dir):
        if candidate is not None and candidate.exists() and any(candidate.iterdir()):
            tokenizer = Tokenizer.load(str(candidate))
            tokenizer.save(str(tokenizer_dir))
            return tokenizer

    tokenizer = Tokenizer()
    if should_train:
        logger.info("Training BPE tokenizer (vocab_size=%d)", vocab_size)
        tokenizer.train([train_text], vocab_size=vocab_size)
        tokenizer.save(str(tokenizer_dir))
        if shared_dir is not None:
            tokenizer.save(str(shared_dir))
        return tokenizer

    msg = "Tokenizer not trained and no path provided"
    raise RuntimeError(msg)


def build_dataloader(
    dataset: PretrainDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, list[int]]]:
    """Create a ``DataLoader`` for pretrain samples."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=make_collate_fn(dataset.pad_id),
    )


def iter_batches(loader: DataLoader[dict[str, list[int]]]) -> Iterator[dict[str, torch.Tensor]]:
    """Yield micro-batches forever, reshuffling each epoch."""
    while True:
        for batch in loader:
            yield batch


def collect_eval_batches(
    dataset: PretrainDataset,
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Materialize all validation micro-batches."""
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False)
    return list(loader)


def is_verbatim_memorization(sample: str, corpus: str, *, min_len: int = 80) -> bool:
    """Return True if ``sample`` contains a long verbatim span from ``corpus``."""
    normalized = " ".join(sample.split())
    if len(normalized) < min_len:
        return False
    for start in range(0, len(normalized) - min_len + 1, 20):
        snippet = normalized[start : start + min_len]
        if snippet in corpus:
            return True
    return False


def run_sampling(
    model: GPT,
    tokenizer: Tokenizer,
    *,
    prompts: list[str],
    temperatures: list[float],
    max_new_tokens: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate samples at multiple temperatures."""
    was_training = model.training
    model.eval()
    outputs: list[dict[str, Any]] = []
    for prompt in prompts:
        for temperature in temperatures:
            torch.manual_seed(seed)
            result = generate(
                model,
                tokenizer,
                prompt,
                GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    sampler=SamplerConfig(temperature=temperature),
                    add_bos=False,
                ),
                device=device,
            )
            outputs.append(
                {
                    "prompt": prompt,
                    "temperature": temperature,
                    "text": result.text,
                }
            )
    if was_training:
        model.train()
    return outputs


def train_from_config(config_path: Path, *, resume: Path | None = None) -> Path:
    """Run training from a YAML config and return the experiment directory."""
    config = load_yaml_config(config_path)
    experiment_section = _require_section(config, "experiment")
    data_section = _require_section(config, "data")
    training_section = _require_section(config, "training")
    eval_section = _require_section(config, "evaluation")
    checkpoint_section = _require_section(config, "checkpoint")
    sampling_section = _require_section(config, "sampling")
    model_section = _require_section(config, "model")

    seed = int(training_section.get("seed", 42))
    torch.manual_seed(seed)

    output_root = ROOT / experiment_section.get("output_dir", "experiments")
    run_dir = next_experiment_dir(output_root)
    copy_config(config_path, run_dir)
    experiment_name = str(experiment_section.get("name", "pretrain"))
    best_weights_path = run_dir / f"{experiment_name}_best.pt"
    final_weights_path = run_dir / f"{experiment_name}.pt"
    logger.info("Experiment directory: %s", run_dir)

    data_dir = ROOT / data_section.get("data_dir", "data/shakespeare")
    val_split = float(data_section.get("val_split", 0.1))
    prepare_shakespeare_splits(data_dir, val_ratio=val_split)
    train_text, val_text = load_split_texts(data_dir)

    shared_tokenizer_dir = data_dir / "tokenizer"
    tokenizer_dir = run_dir / "tokenizer"
    tokenizer = train_or_load_tokenizer(
        config,
        train_text,
        tokenizer_dir,
        shared_dir=shared_tokenizer_dir,
    )
    ctx_len = int(model_section.get("ctx_len", 512))

    train_dataset = PretrainDataset([train_text], tokenizer, ctx_len, seed=seed)
    val_dataset = PretrainDataset([val_text], tokenizer, ctx_len, seed=seed + 1)
    logger.info("Train sequences: %d | Val sequences: %d", len(train_dataset), len(val_dataset))

    gpt_config = build_gpt_config(model_section, tokenizer.vocab_size)
    estimated = estimate_parameter_count(gpt_config)
    logger.info("Model params (estimate): %.2fM", estimated / 1e6)

    trainer_config = TrainerConfig(
        max_steps=int(training_section.get("max_steps", 3000)),
        grad_accum_steps=int(training_section.get("grad_accum_steps", 1)),
        max_grad_norm=float(training_section.get("max_grad_norm", 1.0)),
        log_interval=int(training_section.get("log_interval", 10)),
        eval_interval=int(training_section.get("eval_interval", 50)),
        seed=seed,
        prefer_bf16=bool(training_section.get("prefer_bf16", True)),
        device=training_section.get("device"),
        stage_tag=str(experiment_section.get("name", "pretrain")),
        tokenizer_path=str(tokenizer_dir),
    )
    optimizer_config = AdamWConfig(
        learning_rate=float(training_section.get("learning_rate", 3e-4)),
        weight_decay=float(training_section.get("weight_decay", 0.1)),
    )
    scheduler_config = CosineWarmupConfig(
        warmup_steps=int(training_section.get("warmup_steps", 100)),
        max_steps=trainer_config.max_steps,
    )

    model = GPT(gpt_config)
    logger.info("Model params (actual): %.2fM", count_parameters(model) / 1e6)
    trainer = Trainer(
        model,
        gpt_config,
        trainer_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )

    if resume is not None:
        trainer.load(resume)
        logger.info("Resumed from step %d", trainer.global_step)

    batch_size = int(training_section.get("batch_size", 16))
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_batches = collect_eval_batches(val_dataset, batch_size=batch_size)
    batch_stream = iter_batches(train_loader)

    val_ppl_target = float(eval_section.get("val_perplexity_target", 15.0))
    early_stop = bool(eval_section.get("early_stop", True))
    eval_patience = int(eval_section.get("eval_patience_steps", 200))
    save_interval = int(checkpoint_section.get("save_interval", 500))

    loss_records: list[LossRecord] = []
    latest_train_loss: float | None = None
    latest_eval_loss: float | None = None
    best_eval_loss: float | None = None
    best_step = 0
    steps_since_best = 0
    target_reached = False

    progress = tqdm(
        total=trainer_config.max_steps,
        initial=trainer.global_step,
        desc="train",
        unit="step",
    )

    accum = trainer_config.grad_accum_steps
    while trainer.global_step < trainer_config.max_steps:
        micro_batches = [next(batch_stream) for _ in range(accum)]
        loss, _grad_norm = trainer.optimizer_step_from_micro_batches(micro_batches)
        trainer.global_step += 1
        latest_train_loss = loss
        trainer.metrics.train_losses.append(loss)
        progress.update(1)
        progress.set_postfix(loss=f"{loss:.4f}")

        if trainer.global_step % trainer_config.log_interval == 0:
            lr = trainer.scheduler.get_last_lr()[0]
            logger.info(
                "step=%d train_loss=%.4f lr=%.6f",
                trainer.global_step,
                loss,
                lr,
            )

        should_eval = trainer.global_step % trainer_config.eval_interval == 0
        if should_eval:
            latest_eval_loss = trainer.evaluate(val_batches)
            perplexity = math.exp(latest_eval_loss)
            trainer.metrics.eval_losses.append((trainer.global_step, latest_eval_loss))
            logger.info(
                "step=%d eval_loss=%.4f perplexity=%.2f",
                trainer.global_step,
                latest_eval_loss,
                perplexity,
            )
            progress.set_postfix(loss=f"{loss:.4f}", ppl=f"{perplexity:.1f}")
            if best_eval_loss is None or latest_eval_loss < best_eval_loss:
                best_eval_loss = latest_eval_loss
                best_step = trainer.global_step
                steps_since_best = 0
                trainer.save(best_weights_path)
            else:
                steps_since_best = trainer.global_step - best_step
            if early_stop and perplexity <= val_ppl_target:
                target_reached = True
                logger.info(
                    "Validation perplexity target reached (%.2f <= %.2f)",
                    perplexity,
                    val_ppl_target,
                )
                break
            if early_stop and steps_since_best >= eval_patience:
                logger.info(
                    "Early stopping at step %d (best eval at step %d, ppl=%.2f)",
                    trainer.global_step,
                    best_step,
                    math.exp(best_eval_loss),
                )
                break

        loss_records.append(
            LossRecord(
                step=trainer.global_step,
                train_loss=loss,
                eval_loss=latest_eval_loss if should_eval else None,
            )
        )

        if trainer.global_step % save_interval == 0:
            trainer.save(run_dir / f"{experiment_name}_step_{trainer.global_step}.pt")

    progress.close()

    if best_weights_path.exists():
        trainer.load(best_weights_path)
        latest_eval_loss = best_eval_loss
        logger.info(
            "Restored best checkpoint from step %d (eval_loss=%.4f, ppl=%.2f)",
            best_step,
            best_eval_loss,
            math.exp(best_eval_loss) if best_eval_loss is not None else float("nan"),
        )

    if latest_eval_loss is None:
        latest_eval_loss = trainer.evaluate(val_batches)
    final_perplexity = math.exp(latest_eval_loss)

    trainer.save(final_weights_path)

    write_loss_csv(loss_records, run_dir / "loss.csv")

    prompts = [str(p) for p in sampling_section.get("prompts", ["ROMEO:\n"])]
    temperatures = [float(t) for t in sampling_section.get("temperatures", [0.0, 0.7, 1.0])]
    sample_outputs = run_sampling(
        model,
        tokenizer,
        prompts=prompts,
        temperatures=temperatures,
        max_new_tokens=int(sampling_section.get("max_new_tokens", 120)),
        device=trainer.device,
        seed=int(sampling_section.get("seed", seed)),
    )
    write_samples(run_dir / "samples.txt", sample_outputs)

    memorization_flags = [
        is_verbatim_memorization(entry["text"], train_text)
        for entry in sample_outputs
    ]

    metrics = {
        "experiment": experiment_section.get("name", "run"),
        "steps": trainer.global_step,
        "best_step": best_step,
        "target_reached": target_reached,
        "final_eval_loss": latest_eval_loss,
        "final_perplexity": final_perplexity,
        "best_eval_loss": best_eval_loss,
        "best_perplexity": math.exp(best_eval_loss) if best_eval_loss is not None else None,
        "val_perplexity_target": val_ppl_target,
        "parameter_count": count_parameters(model),
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "memorization_spot_check": {
            "any_verbatim_long_span": any(memorization_flags),
            "per_sample": memorization_flags,
        },
    }
    write_metrics(run_dir / "metrics.json", metrics)
    logger.info(
        "Done — final perplexity=%.2f (target %.2f), weights=%s",
        final_perplexity,
        val_ppl_target,
        final_weights_path,
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run_dir = train_from_config(config_path, resume=args.resume)
    print(f"Experiment saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
