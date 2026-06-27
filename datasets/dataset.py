"""PyTorch Dataset classes for pretrain and supervised fine-tuning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from datasets.cleaning import clean_text
from datasets.masking import build_loss_mask, build_shifted_targets
from datasets.packing import pack_token_streams
from tokenizer.tokenizer import Tokenizer


@dataclass(frozen=True)
class SFTExample:
    """One instruction-tuning example with role fields."""

    system: str
    user: str
    assistant: str


def encode_sft_example(
    tokenizer: Tokenizer,
    example: SFTExample,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> tuple[list[int], int]:
    """Encode an SFT example and return ``(input_ids, prompt_len)``."""
    input_ids: list[int] = []
    if add_bos:
        input_ids.append(tokenizer.vocab.special_token_id("<|bos|>"))

    prompt_text = (
        f"<|system|>{example.system}"
        f"<|user|>{example.user}"
        "<|assistant|>"
    )
    input_ids.extend(tokenizer.encode(prompt_text))
    prompt_len = len(input_ids)

    input_ids.extend(tokenizer.encode(example.assistant))
    if add_eos:
        input_ids.append(tokenizer.vocab.special_token_id("<|eos|>"))

    return input_ids, prompt_len


def build_model_inputs(
    input_ids: list[int],
    *,
    prompt_len: int,
    pad_id: int,
) -> dict[str, list[int]]:
    """Build ``input_ids``, shifted ``targets``, and ``loss_mask`` for one sample."""
    targets = build_shifted_targets(input_ids, pad_id)
    loss_mask = build_loss_mask(prompt_len, len(input_ids))
    return {
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
    }


class PretrainDataset(Dataset[dict[str, list[int]]]):
    """Contiguous packed pretrain sequences with loss on all valid positions."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: Tokenizer,
        ctx_len: int,
        *,
        seed: int = 0,
        clean: bool = True,
    ) -> None:
        """Tokenize, pack, and store fixed-length pretrain blocks."""
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.pad_id = tokenizer.vocab.special_token_id("<|pad|>")
        self.eos_id = tokenizer.vocab.special_token_id("<|eos|>")

        cleaned = [clean_text(text) if clean else text for text in texts]
        tokenized = [
            tokenizer.encode(text, add_bos=True, add_eos=True)
            for text in cleaned
            if text
        ]
        packed = pack_token_streams(tokenized, ctx_len, eos_id=self.eos_id)
        self.samples = [self._to_sample(chunk) for chunk in packed if chunk]

        generator = torch.Generator()
        generator.manual_seed(seed)
        order = torch.randperm(len(self.samples), generator=generator).tolist()
        self.samples = [self.samples[index] for index in order]

    def _to_sample(self, token_ids: list[int]) -> dict[str, list[int]]:
        """Convert a packed chunk into model-ready tensors-as-lists."""
        return build_model_inputs(
            token_ids,
            prompt_len=0,
            pad_id=self.pad_id,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


class SFTDataset(Dataset[dict[str, list[int]]]):
    """Role-formatted instruction dataset with prompt-only loss masking."""

    def __init__(
        self,
        examples: Sequence[SFTExample],
        tokenizer: Tokenizer,
        *,
        seed: int = 0,
        max_len: int | None = None,
    ) -> None:
        """Encode SFT examples and optionally truncate to ``max_len``."""
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.vocab.special_token_id("<|pad|>")
        self.samples: list[dict[str, list[int]]] = []

        for example in examples:
            input_ids, prompt_len = encode_sft_example(tokenizer, example)
            if max_len is not None and len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                prompt_len = min(prompt_len, max_len)
            self.samples.append(
                build_model_inputs(
                    input_ids,
                    prompt_len=prompt_len,
                    pad_id=self.pad_id,
                )
            )

        generator = torch.Generator()
        generator.manual_seed(seed)
        order = torch.randperm(len(self.samples), generator=generator).tolist()
        self.samples = [self.samples[index] for index in order]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


def iter_cleaned_texts(texts: Iterable[str]) -> list[str]:
    """Clean an iterable of raw documents and drop empty results."""
    return [cleaned for text in texts if (cleaned := clean_text(text))]
