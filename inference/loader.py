"""Load a trained GPT checkpoint for inference."""

from __future__ import annotations

from pathlib import Path

import torch

from model.gpt import GPT
from training.checkpoint import gpt_config_from_dict, load_checkpoint
from training.optimizer import AdamWConfig, create_adamw
from training.scheduler import CosineWarmupConfig, create_cosine_warmup_scheduler


def load_gpt_from_checkpoint(checkpoint_path: Path, device: torch.device) -> GPT:
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
