"""Unit tests for the dataset pipeline and loss masking."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.collate import collate_batch, make_collate_fn
from datasets.dataset import PretrainDataset, SFTDataset, SFTExample, encode_sft_example
from datasets.masking import build_loss_mask
from tokenizer import Tokenizer

PRETRAIN_TEXTS = [
    "The examined life is worth living.",
    "Virtue ethics asks what telos guides human flourishing.",
    "Logic separates valid inference from persuasive rhetoric.",
    "Epoché suspends judgment to inspect experience directly.",
]

SFT_EXAMPLES = [
    SFTExample(
        system="You are VERITAS: blunt and decisive.",
        user="Is the unexamined life worth living?",
        assistant="Yes. Socrates was right — without scrutiny, life defaults to drift.",
    ),
    SFTExample(
        system="You are VERITAS.",
        user="Define qualia.",
        assistant="Qualia are the subjective feels of experience — the redness of red.",
    ),
    SFTExample(
        system="You are VERITAS.",
        user="What is dharma?",
        assistant="Duty, cosmic order, and the path a life ought to follow.",
    ),
]


def _sample_tokenizer() -> Tokenizer:
    return Tokenizer.train_sample(vocab_size=512)


def _assert_shift_alignment(input_ids: list[int], targets: list[int]) -> None:
    for position in range(len(input_ids) - 1):
        assert targets[position] == input_ids[position + 1]


def test_build_loss_mask_zeros_prompt_span() -> None:
    """Prompt positions are masked; completion positions contribute to loss."""
    prompt_len = 5
    total_len = 10
    mask = build_loss_mask(prompt_len, total_len)
    assert mask[: prompt_len - 1] == [0] * (prompt_len - 1)
    assert mask[prompt_len - 1 : total_len - 1] == [1] * (total_len - prompt_len)
    assert mask[-1] == 0


def test_build_loss_mask_pretrain_counts_all_positions() -> None:
    """Pretrain masking counts every valid next-token position."""
    mask = build_loss_mask(prompt_len=0, total_len=8)
    assert mask[:-1] == [1] * 7
    assert mask[-1] == 0


def test_sft_loss_mask_zeros_prompt_and_keeps_completion() -> None:
    """SFT samples mask the full prompt span and count assistant tokens."""
    tokenizer = _sample_tokenizer()
    example = SFT_EXAMPLES[0]
    input_ids, prompt_len = encode_sft_example(tokenizer, example)
    loss_mask = build_loss_mask(prompt_len, len(input_ids))

    assert loss_mask[: prompt_len - 1] == [0] * (prompt_len - 1)
    assert any(loss_mask[position] == 1 for position in range(prompt_len - 1, len(input_ids) - 1))
    assert loss_mask[-1] == 0


def test_targets_match_shifted_input_ids() -> None:
    """targets[t] == input_ids[t + 1] for every valid position."""
    tokenizer = _sample_tokenizer()
    pad_id = tokenizer.vocab.special_token_id("<|pad|>")

    pretrain = PretrainDataset(PRETRAIN_TEXTS, tokenizer, ctx_len=32, seed=0)
    sample = pretrain[0]
    _assert_shift_alignment(sample["input_ids"], sample["targets"])
    assert sample["targets"][-1] == pad_id

    sft = SFTDataset(SFT_EXAMPLES, tokenizer, seed=0)
    sft_sample = sft[0]
    _assert_shift_alignment(sft_sample["input_ids"], sft_sample["targets"])


def test_padding_positions_excluded_from_loss() -> None:
    """Collator padding never contributes to the loss."""
    tokenizer = _sample_tokenizer()
    pad_id = tokenizer.vocab.special_token_id("<|pad|>")
    dataset = SFTDataset(SFT_EXAMPLES, tokenizer, seed=0)

    batch = collate_batch([dataset[0], dataset[1]], pad_id=pad_id)
    pad_rows, pad_cols = torch.where(batch["input_ids"] == pad_id)
    for row, col in zip(pad_rows.tolist(), pad_cols.tolist(), strict=True):
        assert batch["loss_mask"][row, col].item() == 0
        assert batch["targets"][row, col].item() == pad_id


def test_collate_batch_shapes() -> None:
    """Collated batch tensors share shape [B, T]."""
    tokenizer = _sample_tokenizer()
    pad_id = tokenizer.vocab.special_token_id("<|pad|>")
    dataset = SFTDataset(SFT_EXAMPLES, tokenizer, seed=0)
    batch = collate_batch([dataset[i] for i in range(3)], pad_id=pad_id)

    batch_size = 3
    max_len = max(len(dataset[i]["input_ids"]) for i in range(3))
    assert batch["input_ids"].shape == (batch_size, max_len)
    assert batch["targets"].shape == (batch_size, max_len)
    assert batch["loss_mask"].shape == (batch_size, max_len)


def test_deterministic_batches_with_fixed_seed() -> None:
    """Fixed seed yields identical batches across DataLoader runs."""
    tokenizer = _sample_tokenizer()
    pad_id = tokenizer.vocab.special_token_id("<|pad|>")

    def run_loader(seed: int) -> dict[str, torch.Tensor]:
        dataset = SFTDataset(SFT_EXAMPLES, tokenizer, seed=seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=make_collate_fn(pad_id),
            generator=generator,
        )
        return next(iter(loader))

    first = run_loader(seed=42)
    second = run_loader(seed=42)
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["targets"], second["targets"])
    assert torch.equal(first["loss_mask"], second["loss_mask"])


def test_pretrain_dataset_loss_on_all_valid_positions() -> None:
    """Pretrain samples apply loss to every valid next-token position."""
    tokenizer = _sample_tokenizer()
    dataset = PretrainDataset(PRETRAIN_TEXTS, tokenizer, ctx_len=24, seed=7)
    sample = dataset[0]
    assert sample["loss_mask"][:-1] == [1] * (len(sample["input_ids"]) - 1)
    assert sample["loss_mask"][-1] == 0


def test_pretrain_dataset_deterministic_order() -> None:
    """Pretrain shuffling is reproducible under a fixed seed."""
    tokenizer = _sample_tokenizer()
    first = PretrainDataset(PRETRAIN_TEXTS, tokenizer, ctx_len=24, seed=99)
    second = PretrainDataset(PRETRAIN_TEXTS, tokenizer, ctx_len=24, seed=99)
    assert first[0]["input_ids"] == second[0]["input_ids"]
    assert first[-1]["input_ids"] == second[-1]["input_ids"]
