"""Unit tests for general corpus mixing and shard helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from datasets.shards import save_token_shards
from datasets.sources.general import (
    CorpusMixConfig,
    _load_documents_from_parquet_urls,
    _relative_parquet_path,
    merge_validation_documents,
    pack_documents_to_chunks,
    weighted_train_documents,
)
from tokenizer import Tokenizer


def test_relative_parquet_path_parses_hf_urls() -> None:
    """Parquet cache paths are derived from resolve URLs."""
    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )
    assert _relative_parquet_path(url) == "train/0000.parquet"


def test_load_documents_from_local_parquet(tmp_path: Path) -> None:
    """Parquet loader reads the text column from on-disk shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "parquet" / "train" / "0000.parquet"
    parquet_path.parent.mkdir(parents=True)
    table = pa.table({"text": ["alpha story", "beta story"]})
    pq.write_table(table, parquet_path)

    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )

    docs = _load_documents_from_parquet_urls([url], tmp_path, max_docs=1)
    assert docs == ["alpha story"]


def test_weighted_train_documents_upweights_wikitext() -> None:
    """WikiText repeats according to ``wikitext_weight``."""
    mix = CorpusMixConfig(
        wikitext_weight=3.0,
        tinystories_weight=1.0,
        openwebtext_weight=1.0,
        seed=7,
    )
    mixed = weighted_train_documents(
        wikitext=["wiki-a", "wiki-b"],
        tinystories=["story-a"],
        openwebtext=["web-a"],
        mix=mix,
    )
    assert mixed.count("wiki-a") == 3
    assert mixed.count("wiki-b") == 3
    assert mixed.count("story-a") == 1
    assert mixed.count("web-a") == 1


def test_merge_validation_documents_concatenates_sources() -> None:
    """Validation docs from all sources are retained."""
    merged = merge_validation_documents(
        wikitext_val=["w1"],
        tinystories_val=["t1", "t2"],
        openwebtext_val=["o1"],
    )
    assert merged == ["w1", "t1", "t2", "o1"]


def test_pack_documents_to_chunks_respects_ctx_len() -> None:
    """Packed chunks never exceed ``ctx_len``."""
    tokenizer = Tokenizer.train_sample(vocab_size=512)
    chunks = pack_documents_to_chunks(
        ["alpha beta gamma", "delta epsilon zeta eta theta"],
        tokenizer,
        ctx_len=16,
    )
    assert chunks
    assert all(len(chunk) <= 16 for chunk in chunks)


def test_save_token_shards_round_trip(tmp_path: Path) -> None:
    """Saved shards can be reloaded from disk."""
    chunks = [[1, 2, 3], [4, 5, 6, 7]]
    output_dir = tmp_path / "train"
    paths = save_token_shards(chunks, output_dir, shard_size=1)
    assert len(paths) == 2
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert payload["chunks"] == [[1, 2, 3]]
