"""Project LOGOS — dataset classes, chunkers, and batch collators."""

from datasets.chunking import chunk_token_ids
from datasets.cleaning import clean_text
from datasets.collate import collate_batch, make_collate_fn
from datasets.dataset import (
    PretrainDataset,
    SFTDataset,
    SFTExample,
    build_model_inputs,
    encode_sft_example,
)
from datasets.masking import (
    apply_padding_to_loss_mask,
    build_loss_mask,
    build_pretrain_loss_mask,
    build_shifted_targets,
)
from datasets.packing import pack_token_streams

__all__ = [
    "PretrainDataset",
    "SFTDataset",
    "SFTExample",
    "apply_padding_to_loss_mask",
    "build_loss_mask",
    "build_model_inputs",
    "build_pretrain_loss_mask",
    "build_shifted_targets",
    "chunk_token_ids",
    "clean_text",
    "collate_batch",
    "encode_sft_example",
    "make_collate_fn",
    "pack_token_streams",
]
