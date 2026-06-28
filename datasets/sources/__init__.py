"""Remote and local corpus source loaders."""

from datasets.sources.general import (
    CorpusMixConfig,
    prepare_general_shards,
    weighted_train_documents,
)

__all__ = [
    "CorpusMixConfig",
    "prepare_general_shards",
    "weighted_train_documents",
]
