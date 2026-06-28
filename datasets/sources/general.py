"""General pretraining corpus loaders, mixing, and shard preparation."""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.cleaning import clean_text
from datasets.packing import pack_token_streams
from datasets.shards import list_shard_paths, save_token_shards
from tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

HF_PARQUET_API = "https://datasets-server.huggingface.co/parquet"
WIKITEXT_PARQUET_BASE = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    "refs%2Fconvert%2Fparquet/wikitext-103-raw-v1"
)
WIKITEXT_PARQUET_FILES: dict[str, list[str]] = {
    "train": ["train/0000.parquet", "train/0001.parquet"],
    "validation": ["validation/0000.parquet"],
    "test": ["test/0000.parquet"],
}
TINYSTORIES_DATASET = "roneneldan/TinyStories"
OPENWEBTEXT_DATASET = "Skylion007/openwebtext"
OPENWEBTEXT_CONFIG = "plain_text"
HF_RETRY_ATTEMPTS = 10
HF_RETRY_BASE_DELAY = 2.0
HF_HTTP_TIMEOUT = 180.0
PARQUET_PATH_MARKERS = ("/default/", "/plain_text/", "/wikitext-103-raw-v1/")


@dataclass(frozen=True)
class CorpusMixConfig:
    """Per-source sampling limits and train mix weights."""

    wikitext_weight: float = 5.0
    tinystories_weight: float = 1.0
    openwebtext_weight: float = 1.0
    wikitext_train_max: int | None = 100_000
    tinystories_train_max: int = 50_000
    tinystories_val_max: int = 2_000
    openwebtext_train_max: int = 10_000
    openwebtext_val_max: int = 1_000
    shard_size: int = 512
    seed: int = 42


@dataclass(frozen=True)
class PreparedCorpus:
    """Paths and stats for a prepared general corpus."""

    data_dir: Path
    train_shard_dir: Path
    val_shard_dir: Path
    train_sequences: int
    val_sequences: int
    train_documents: int
    val_documents: int


def build_mix_config(data_section: dict[str, Any]) -> CorpusMixConfig:
    """Build ``CorpusMixConfig`` from a YAML ``data.mix`` section."""
    mix_section = data_section.get("mix", {})
    if not isinstance(mix_section, dict):
        mix_section = {}
    seed = int(data_section.get("seed", mix_section.get("seed", 42)))
    return CorpusMixConfig(
        wikitext_weight=float(mix_section.get("wikitext_weight", 5.0)),
        tinystories_weight=float(mix_section.get("tinystories_weight", 1.0)),
        openwebtext_weight=float(mix_section.get("openwebtext_weight", 1.0)),
        wikitext_train_max=(
            None
            if mix_section.get("wikitext_train_max") is None
            else int(mix_section["wikitext_train_max"])
        ),
        tinystories_train_max=int(mix_section.get("tinystories_train_max", 50_000)),
        tinystories_val_max=int(mix_section.get("tinystories_val_max", 2_000)),
        openwebtext_train_max=int(mix_section.get("openwebtext_train_max", 10_000)),
        openwebtext_val_max=int(mix_section.get("openwebtext_val_max", 1_000)),
        shard_size=int(mix_section.get("shard_size", 512)),
        seed=seed,
    )


def _http_get_json(url: str, *, timeout: float = HF_HTTP_TIMEOUT) -> dict[str, object]:
    """Fetch JSON from ``url``."""
    request = urllib.request.Request(url, headers={"User-Agent": "logos/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object from {url}"
        raise TypeError(msg)
    return payload


def _http_get_json_with_retry(url: str, *, timeout: float = HF_HTTP_TIMEOUT) -> dict[str, object]:
    """Fetch JSON with exponential backoff on HTTP 429."""
    delay = HF_RETRY_BASE_DELAY
    last_error: urllib.error.HTTPError | None = None
    for attempt in range(HF_RETRY_ATTEMPTS):
        try:
            return _http_get_json(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt + 1 < HF_RETRY_ATTEMPTS:
                logger.warning("HF rate limit (429); retrying in %.1fs …", delay)
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            raise
        except TimeoutError:
            if attempt + 1 < HF_RETRY_ATTEMPTS:
                logger.warning("HF request timed out; retrying in %.1fs …", delay)
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            raise
    if last_error is not None:
        raise last_error
    msg = f"Failed to fetch JSON from {url}"
    raise RuntimeError(msg)


def _http_download(url: str, destination: Path, *, timeout: float = 120.0) -> None:
    """Download ``url`` to ``destination`` if missing, following HTTP redirects."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "logos/0.1"})
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        destination.write_bytes(response.read())


def _save_doc_cache(path: Path, documents: list[str]) -> None:
    """Persist one JSON-encoded document per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def _load_doc_cache(path: Path, *, max_docs: int | None = None) -> list[str]:
    """Load cached documents written by ``_save_doc_cache``."""
    documents: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if max_docs is not None and len(documents) >= max_docs:
                break
            documents.append(json.loads(line))
    return documents


def _relative_parquet_path(url: str) -> str:
    """Extract ``split/file.parquet`` from a HuggingFace resolve URL."""
    for marker in PARQUET_PATH_MARKERS:
        if marker in url:
            return url.split(marker, 1)[1]
    msg = f"Cannot parse parquet path from {url}"
    raise ValueError(msg)


def _fetch_parquet_url_index(
    dataset: str,
    *,
    config: str = "default",
    cache_path: Path | None = None,
) -> dict[str, list[str]]:
    """Return parquet download URLs grouped by split."""
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(key): list(value) for key, value in payload.items() if isinstance(value, list)}

    api_url = f"{HF_PARQUET_API}?dataset={dataset}"
    if config != "default":
        api_url += f"&config={config}"
    payload = _http_get_json_with_retry(api_url)
    files = payload.get("parquet_files")
    if not isinstance(files, list):
        msg = f"Missing parquet_files for dataset {dataset}"
        raise TypeError(msg)

    index: dict[str, list[str]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        split = str(entry.get("split", "train"))
        file_url = entry.get("url")
        if isinstance(file_url, str):
            index.setdefault(split, []).append(file_url)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def _download_parquet_url(url: str, cache_dir: Path) -> Path:
    """Download one parquet shard if missing."""
    relative_path = _relative_parquet_path(url)
    destination = cache_dir / "parquet" / relative_path
    if destination.exists():
        return destination
    logger.info("Downloading parquet %s …", relative_path)
    _http_download(url, destination, timeout=600.0)
    return destination


def _load_documents_from_parquet_urls(
    urls: list[str],
    cache_dir: Path,
    *,
    max_docs: int | None = None,
) -> list[str]:
    """Stream documents from parquet shards until ``max_docs`` is reached."""
    if max_docs is not None and max_docs <= 0:
        return []

    documents: list[str] = []
    for url in urls:
        remaining = None
        if max_docs is not None:
            remaining = max_docs - len(documents)
            if remaining <= 0:
                break
        parquet_path = _download_parquet_url(url, cache_dir)
        documents.extend(_load_parquet_text_column(parquet_path, max_docs=remaining))
    return documents


def _wikitext_parquet_url(relative_path: str) -> str:
    """Return the HuggingFace resolve URL for a WikiText parquet shard."""
    return f"{WIKITEXT_PARQUET_BASE}/{relative_path}"


def _download_wikitext_parquet(relative_path: str, cache_dir: Path) -> Path:
    """Download one WikiText parquet shard if missing."""
    destination = cache_dir / "parquet" / relative_path
    if destination.exists():
        return destination
    logger.info("Downloading WikiText parquet %s …", relative_path)
    _http_download(
        _wikitext_parquet_url(relative_path),
        destination,
        timeout=600.0,
    )
    return destination


def _load_parquet_text_column(path: Path, *, max_docs: int | None = None) -> list[str]:
    """Read the ``text`` column from a parquet shard."""
    import pyarrow.parquet as pq

    if max_docs is not None and max_docs <= 0:
        return []

    parquet_file = pq.ParquetFile(path)
    documents: list[str] = []
    for batch in parquet_file.iter_batches(batch_size=10_000, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if isinstance(text, str) and text.strip():
                documents.append(clean_text(text))
                if max_docs is not None and len(documents) >= max_docs:
                    return documents
    return documents


def _load_wikitext_split_from_parquet(
    split: str,
    cache_dir: Path,
    *,
    max_docs: int | None = None,
) -> list[str]:
    """Load one WikiText split from cached or downloaded parquet files."""
    documents: list[str] = []
    for relative_path in WIKITEXT_PARQUET_FILES[split]:
        remaining = None
        if max_docs is not None:
            remaining = max_docs - len(documents)
            if remaining <= 0:
                break
        parquet_path = _download_wikitext_parquet(relative_path, cache_dir)
        documents.extend(
            _load_parquet_text_column(parquet_path, max_docs=remaining),
        )
    return documents


def _wikitext_train_cache_path(cache_dir: Path, train_max: int | None) -> Path:
    """Return the on-disk cache path for a WikiText train subset."""
    if train_max is None:
        return cache_dir / "train.jsonl"
    return cache_dir / f"train.max{train_max}.jsonl"


def load_wikitext_documents(
    data_dir: str | Path,
    *,
    train_max: int | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(train_docs, val_docs)`` from WikiText-103 parquet (cached on disk)."""
    cache_dir = Path(data_dir) / "wikitext"
    train_cache = _wikitext_train_cache_path(cache_dir, train_max)
    val_cache = cache_dir / "val.jsonl"

    if val_cache.exists():
        val_docs = _load_doc_cache(val_cache)
    else:
        logger.info("Loading WikiText-103 validation + test splits from parquet …")
        val_docs = _load_wikitext_split_from_parquet("validation", cache_dir)
        val_docs.extend(_load_wikitext_split_from_parquet("test", cache_dir))
        _save_doc_cache(val_cache, val_docs)

    if train_cache.exists():
        train_docs = _load_doc_cache(train_cache, max_docs=train_max)
        return train_docs, val_docs

    logger.info("Loading WikiText-103 train split from parquet …")
    train_docs = _load_wikitext_split_from_parquet("train", cache_dir, max_docs=train_max)
    _save_doc_cache(train_cache, train_docs)
    return train_docs, val_docs


def load_tinystories_documents(
    mix: CorpusMixConfig,
    *,
    data_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(train_docs, val_docs)`` from TinyStories parquet shards."""
    cache_dir = Path(data_dir or "data/general") / "tinystories"
    train_cache = cache_dir / f"train.max{mix.tinystories_train_max}.jsonl"
    val_cache = cache_dir / f"val.max{mix.tinystories_val_max}.jsonl"
    url_index = _fetch_parquet_url_index(
        TINYSTORIES_DATASET,
        cache_path=cache_dir / "parquet_urls.json",
    )

    if train_cache.exists():
        train_docs = _load_doc_cache(train_cache)
    else:
        logger.info(
            "Loading TinyStories train split from parquet (max %d docs) …",
            mix.tinystories_train_max,
        )
        train_docs = _load_documents_from_parquet_urls(
            url_index.get("train", []),
            cache_dir,
            max_docs=mix.tinystories_train_max,
        )
        _save_doc_cache(train_cache, train_docs)

    if val_cache.exists():
        val_docs = _load_doc_cache(val_cache)
    else:
        logger.info(
            "Loading TinyStories validation split from parquet (max %d docs) …",
            mix.tinystories_val_max,
        )
        val_docs = _load_documents_from_parquet_urls(
            url_index.get("validation", []),
            cache_dir,
            max_docs=mix.tinystories_val_max,
        )
        _save_doc_cache(val_cache, val_docs)

    return train_docs, val_docs


def load_openwebtext_documents(
    mix: CorpusMixConfig,
    *,
    data_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(train_docs, val_docs)`` from an OpenWebText parquet subset."""
    cache_dir = Path(data_dir or "data/general") / "openwebtext"
    total_rows = mix.openwebtext_train_max + mix.openwebtext_val_max
    combined_cache = cache_dir / f"train.max{total_rows}.jsonl"

    if combined_cache.exists():
        all_docs = _load_doc_cache(combined_cache)
    else:
        url_index = _fetch_parquet_url_index(
            OPENWEBTEXT_DATASET,
            config=OPENWEBTEXT_CONFIG,
            cache_path=cache_dir / "parquet_urls.json",
        )
        logger.info(
            "Loading OpenWebText train split from parquet (max %d docs) …",
            total_rows,
        )
        all_docs = _load_documents_from_parquet_urls(
            url_index.get("train", []),
            cache_dir,
            max_docs=total_rows,
        )
        _save_doc_cache(combined_cache, all_docs)

    if len(all_docs) <= mix.openwebtext_val_max:
        return all_docs, []
    split_idx = len(all_docs) - mix.openwebtext_val_max
    return all_docs[:split_idx], all_docs[split_idx:]


def weighted_train_documents(
    *,
    wikitext: list[str],
    tinystories: list[str],
    openwebtext: list[str],
    mix: CorpusMixConfig,
) -> list[str]:
    """Repeat WikiText more often so factual/expository text dominates the mix."""
    weighted: list[str] = []
    source_map = {
        "wikitext": (wikitext, mix.wikitext_weight),
        "tinystories": (tinystories, mix.tinystories_weight),
        "openwebtext": (openwebtext, mix.openwebtext_weight),
    }
    for name, (docs, weight) in source_map.items():
        if not docs or weight <= 0:
            continue
        repeats = max(1, int(round(weight)))
        for _ in range(repeats):
            weighted.extend(docs)
        logger.info(
            "Mix %s: %d docs × %d repeats = %d train documents",
            name,
            len(docs),
            repeats,
            len(docs) * repeats,
        )

    rng = random.Random(mix.seed)
    rng.shuffle(weighted)
    return weighted


def merge_validation_documents(
    *,
    wikitext_val: list[str],
    tinystories_val: list[str],
    openwebtext_val: list[str],
) -> list[str]:
    """Combine held-out documents from each source."""
    return wikitext_val + tinystories_val + openwebtext_val


def tokenizer_training_corpus(documents: Iterable[str], *, max_chars: int = 8_000_000) -> str:
    """Build a bounded text sample for BPE training."""
    parts: list[str] = []
    total = 0
    for document in documents:
        if total >= max_chars:
            break
        parts.append(document)
        total += len(document) + 1
    return "\n".join(parts)


def pack_documents_to_chunks(
    documents: list[str],
    tokenizer: Tokenizer,
    ctx_len: int,
) -> list[list[int]]:
    """Tokenize and pack documents into fixed-length chunks."""
    eos_id = tokenizer.vocab.special_token_id("<|eos|>")
    tokenized = [
        tokenizer.encode(document, add_bos=True, add_eos=True)
        for document in documents
        if document
    ]
    return pack_token_streams(tokenized, ctx_len, eos_id=eos_id)


def prepare_general_shards(
    data_dir: str | Path,
    tokenizer: Tokenizer,
    ctx_len: int,
    mix: CorpusMixConfig,
    *,
    force: bool = False,
) -> PreparedCorpus:
    """Download sources, mix, tokenize, and write train/val shards at ``ctx_len``."""
    root = Path(data_dir)
    train_dir = root / "shards" / "train"
    val_dir = root / "shards" / "val"
    meta_path = root / "shards" / "meta.json"

    if not force and list_shard_paths(train_dir) and list_shard_paths(val_dir):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return PreparedCorpus(
            data_dir=root,
            train_shard_dir=train_dir,
            val_shard_dir=val_dir,
            train_sequences=int(meta["train_sequences"]),
            val_sequences=int(meta["val_sequences"]),
            train_documents=int(meta["train_documents"]),
            val_documents=int(meta["val_documents"]),
        )

    wiki_train, wiki_val = load_wikitext_documents(root, train_max=mix.wikitext_train_max)
    ts_train, ts_val = load_tinystories_documents(mix, data_dir=root)
    owt_train, owt_val = load_openwebtext_documents(mix, data_dir=root)

    train_docs = weighted_train_documents(
        wikitext=wiki_train,
        tinystories=ts_train,
        openwebtext=owt_train,
        mix=mix,
    )
    val_docs = merge_validation_documents(
        wikitext_val=wiki_val,
        tinystories_val=ts_val,
        openwebtext_val=owt_val,
    )

    logger.info("Packing %d train docs and %d val docs at ctx_len=%d", len(train_docs), len(val_docs), ctx_len)
    train_chunks = pack_documents_to_chunks(train_docs, tokenizer, ctx_len)
    val_chunks = pack_documents_to_chunks(val_docs, tokenizer, ctx_len)

    save_token_shards(train_chunks, train_dir, shard_size=mix.shard_size)
    save_token_shards(val_chunks, val_dir, shard_size=mix.shard_size)

    meta = {
        "ctx_len": ctx_len,
        "mix": mix.__dict__,
        "train_sequences": len(train_chunks),
        "val_sequences": len(val_chunks),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return PreparedCorpus(
        data_dir=root,
        train_shard_dir=train_dir,
        val_shard_dir=val_dir,
        train_sequences=len(train_chunks),
        val_sequences=len(val_chunks),
        train_documents=len(train_docs),
        val_documents=len(val_docs),
    )


def iter_source_documents(data_dir: str | Path, mix: CorpusMixConfig) -> Iterator[str]:
    """Yield raw train documents from all sources (unweighted) for diagnostics."""
    wiki_train, _ = load_wikitext_documents(data_dir, train_max=mix.wikitext_train_max)
    ts_train, _ = load_tinystories_documents(mix, data_dir=data_dir)
    owt_train, _ = load_openwebtext_documents(mix, data_dir=data_dir)
    yield from wiki_train
    yield from ts_train
    yield from owt_train
