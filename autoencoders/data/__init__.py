"""Utilities for working with real embedding matrices."""

from .base import (
    AutoencoderDataset,
    BaseDatasetConfig,
    CachedDataset,
    DataSpec,
    DictSpec,
    DatasetLoaders,
    DatasetSplits,
    ListSpec,
    TensorSpec,
    create_dataloaders,
    default_cache_dir,
    split_dataset,
)
from .clip import (
    CLIPBackedDataset,
    CLIPBackedDatasetConfig,
    CLIPEmbeddingEncoder,
    CLIPRecord,
    OpenCLIPEmbeddingEncoder,
)
from .embeddings import (
    EmbeddingMatrix,
    EmbeddingTensorDataset,
    load_embedding_artifact,
    load_text_embedding_matrix,
    save_embedding_artifact,
)
from .fasttext import FastTextEnglishDataset, FastTextEnglishDatasetConfig
from .flickr30k import Flickr30kDataset, Flickr30kDatasetConfig
from .glove import GloVeDataset, GloVeDatasetConfig
from .loading import get_dataset_class, get_dataset_modules, load_dataset
from .multinli import MultiNLIDataset, MultiNLIDatasetConfig
from .numberbatch import ConceptNetNumberbatchDataset, ConceptNetNumberbatchDatasetConfig
from .snli import SNLIDataset, SNLIDatasetConfig
from .text import (
    EncoderBackedTextDataset,
    EncoderBackedTextDatasetConfig,
    SentenceTransformerTextEncoder,
    TextEmbeddingEncoder,
    TextEmbeddingExample,
)

__all__ = [
    "AutoencoderDataset",
    "BaseDatasetConfig",
    "CachedDataset",
    "CLIPBackedDataset",
    "CLIPBackedDatasetConfig",
    "CLIPEmbeddingEncoder",
    "CLIPRecord",
    "ConceptNetNumberbatchDatasetConfig",
    "DataSpec",
    "DatasetLoaders",
    "DatasetSplits",
    "DictSpec",
    "EmbeddingMatrix",
    "EmbeddingTensorDataset",
    "EncoderBackedTextDataset",
    "EncoderBackedTextDatasetConfig",
    "FastTextEnglishDataset",
    "FastTextEnglishDatasetConfig",
    "Flickr30kDataset",
    "Flickr30kDatasetConfig",
    "GloVeDataset",
    "GloVeDatasetConfig",
    "ConceptNetNumberbatchDataset",
    "ListSpec",
    "MultiNLIDataset",
    "MultiNLIDatasetConfig",
    "OpenCLIPEmbeddingEncoder",
    "SentenceTransformerTextEncoder",
    "SNLIDataset",
    "SNLIDatasetConfig",
    "TensorSpec",
    "TextEmbeddingEncoder",
    "TextEmbeddingExample",
    "create_dataloaders",
    "default_cache_dir",
    "get_dataset_class",
    "get_dataset_modules",
    "load_dataset",
    "load_embedding_artifact",
    "load_text_embedding_matrix",
    "save_embedding_artifact",
    "split_dataset",
]
