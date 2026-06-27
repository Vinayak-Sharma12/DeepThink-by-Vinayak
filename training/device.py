"""Device selection and mixed-precision dtype policy."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Select CPU, MPS, or CUDA based on availability and an optional override."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_bf16(device: torch.device) -> bool:
    """Return True when ``device`` can run bfloat16 training kernels."""
    if device.type == "cuda":
        return bool(torch.cuda.is_bf16_supported())
    return device.type == "mps"


def resolve_dtype(device: torch.device, *, prefer_bf16: bool = True) -> torch.dtype:
    """Pick bf16 on CUDA/MPS when supported; otherwise fp32 (always on CPU)."""
    if device.type == "cpu":
        return torch.float32
    if prefer_bf16 and supports_bf16(device):
        return torch.bfloat16
    return torch.float32


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move collated batch tensors to ``device``."""
    return {key: value.to(device) for key, value in batch.items()}
