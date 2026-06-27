"""Batch collation for dataset samples."""

from __future__ import annotations

from typing import Any

import torch

from datasets.masking import build_shifted_targets


def collate_batch(
    batch: list[dict[str, list[int]]],
    *,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch and return ``(input_ids, targets, loss_mask)`` tensors.

    Padding positions receive ``pad_id`` in ``input_ids`` and ``targets``, and
    their ``loss_mask`` entries are forced to zero.
    """
    if not batch:
        msg = "Cannot collate an empty batch"
        raise ValueError(msg)

    max_len = max(len(sample["input_ids"]) for sample in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for row, sample in enumerate(batch):
        sample_len = len(sample["input_ids"])
        input_ids[row, :sample_len] = torch.tensor(sample["input_ids"], dtype=torch.long)
        if "targets" in sample:
            targets[row, :sample_len] = torch.tensor(sample["targets"], dtype=torch.long)
        else:
            shifted = build_shifted_targets(sample["input_ids"], pad_id)
            targets[row, :sample_len] = torch.tensor(shifted, dtype=torch.long)
        loss_mask[row, :sample_len] = torch.tensor(sample["loss_mask"], dtype=torch.long)

        pad_positions = input_ids[row] == pad_id
        loss_mask[row, pad_positions] = 0

    return {
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
    }


def make_collate_fn(pad_id: int) -> Any:
    """Return a ``collate_fn`` closure for ``DataLoader``."""
    def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return collate_batch(batch, pad_id=pad_id)

    return collate_fn
