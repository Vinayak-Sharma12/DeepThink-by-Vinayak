"""Experiment run directories, metrics export, and sample logging."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LossRecord:
    """One row in the training loss curve."""

    step: int
    train_loss: float | None = None
    eval_loss: float | None = None

    @property
    def perplexity(self) -> float | None:
        """Return ``exp(eval_loss)`` when eval loss is present."""
        if self.eval_loss is None:
            return None
        return math.exp(self.eval_loss)


def next_experiment_dir(base_dir: str | Path) -> Path:
    """Allocate ``experiments/experiment_NNN`` under ``base_dir``."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("experiment_"):
            continue
        suffix = path.name.removeprefix("experiment_")
        if suffix.isdigit():
            numbers.append(int(suffix))
    next_index = max(numbers, default=0) + 1
    run_dir = root / f"experiment_{next_index:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def copy_config(config_path: str | Path, run_dir: str | Path) -> Path:
    """Copy the YAML config into the experiment directory."""
    source = Path(config_path)
    destination = Path(run_dir) / "config.yaml"
    shutil.copy2(source, destination)
    return destination


def write_loss_csv(records: list[LossRecord], path: str | Path) -> None:
    """Write ``loss.csv`` with step, train_loss, eval_loss, perplexity."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "train_loss", "eval_loss", "perplexity"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "step": record.step,
                    "train_loss": record.train_loss,
                    "eval_loss": record.eval_loss,
                    "perplexity": record.perplexity,
                }
            )


def write_samples(path: str | Path, samples: list[dict[str, Any]]) -> None:
    """Write temperature-varied generation samples to a text file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for entry in samples:
        lines.append(f"=== prompt: {entry['prompt']!r} ===")
        lines.append(f"temperature: {entry['temperature']}")
        lines.append(entry["text"])
        lines.append("")
    destination.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Persist final run metrics as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML training config."""
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        msg = f"Config must be a mapping, got {type(loaded).__name__}"
        raise TypeError(msg)
    return loaded
