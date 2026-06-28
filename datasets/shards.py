"""Save and load packed pretrain token shards."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets.dataset import build_model_inputs
from tokenizer.tokenizer import Tokenizer


def save_token_shards(
    chunks: list[list[int]],
    output_dir: str | Path,
    *,
    shard_size: int = 512,
) -> list[Path]:
    """Write ``chunks`` to ``shard_XXX.pt`` files under ``output_dir``."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for existing in root.glob("shard_*.pt"):
        existing.unlink()

    paths: list[Path] = []
    if not chunks:
        return paths

    for index, start in enumerate(range(0, len(chunks), shard_size)):
        batch = chunks[start : start + shard_size]
        path = root / f"shard_{index:03d}.pt"
        torch.save({"chunks": batch}, path)
        paths.append(path)
    return paths


def list_shard_paths(shard_dir: str | Path) -> list[Path]:
    """Return sorted shard file paths under ``shard_dir``."""
    root = Path(shard_dir)
    if not root.exists():
        return []
    return sorted(root.glob("shard_*.pt"))


def load_shard_chunks(path: str | Path) -> list[list[int]]:
    """Load token chunks from one shard file."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        msg = f"Shard {path} missing a list under 'chunks'"
        raise TypeError(msg)
    return chunks


class ShardPretrainDataset(Dataset[dict[str, list[int]]]):
    """Pretrain dataset backed by on-disk packed token shards."""

    def __init__(
        self,
        shard_dir: str | Path,
        tokenizer: Tokenizer,
        *,
        seed: int = 0,
    ) -> None:
        """Load all shards in ``shard_dir`` and expose packed samples."""
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.vocab.special_token_id("<|pad|>")
        shard_paths = list_shard_paths(shard_dir)
        if not shard_paths:
            msg = f"No shard files found in {shard_dir}"
            raise FileNotFoundError(msg)

        samples: list[dict[str, list[int]]] = []
        for path in shard_paths:
            for chunk in load_shard_chunks(path):
                samples.append(
                    build_model_inputs(
                        chunk,
                        prompt_len=0,
                        pad_id=self.pad_id,
                    )
                )

        generator = torch.Generator()
        generator.manual_seed(seed)
        order = torch.randperm(len(samples), generator=generator).tolist()
        self.samples = [samples[index] for index in order]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]
