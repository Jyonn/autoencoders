"""Utilities for working with real embedding matrices."""

from .base import (
    AutoencoderDataset,
    CachedDataset,
    DatasetLoaders,
    DatasetSplits,
    create_dataloaders,
    default_cache_dir,
    split_dataset,
)
from .embeddings import (
    EmbeddingMatrix,
    EmbeddingTensorDataset,
    load_embedding_artifact,
    load_text_embedding_matrix,
    save_embedding_artifact,
)
from .fasttext import FastTextEnglishDataset
from .glove import GloVeDataset
from .loading import load_dataset
from .numberbatch import ConceptNetNumberbatchDataset

__all__ = [
    "AutoencoderDataset",
    "CachedDataset",
    "DatasetLoaders",
    "DatasetSplits",
    "EmbeddingMatrix",
    "EmbeddingTensorDataset",
    "FastTextEnglishDataset",
    "GloVeDataset",
    "ConceptNetNumberbatchDataset",
    "create_dataloaders",
    "default_cache_dir",
    "load_dataset",
    "load_embedding_artifact",
    "load_text_embedding_matrix",
    "save_embedding_artifact",
    "split_dataset",
]
