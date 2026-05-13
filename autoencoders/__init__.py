"""Top-level package for the autoencoders library."""

from .configuration_utils import PretrainedConfig
from .modeling_outputs import AutoencoderOutput
from .models.ae.configuration_ae import AutoencoderConfig
from .models.base.configuration_base import BaseAutoencoderConfig

__all__ = [
    "AutoencoderConfig",
    "AutoencoderOutput",
    "BaseAutoencoderConfig",
    "PretrainedConfig",
]

try:
    from .modeling_utils import PreTrainedAutoencoderModel
    from .data import (
        AutoencoderDataset,
        CachedDataset,
        DatasetLoaders,
        DatasetSplits,
        EmbeddingMatrix,
        EmbeddingTensorDataset,
        GloVeDataset,
        create_dataloaders,
        load_embedding_artifact,
        load_text_embedding_matrix,
        split_dataset,
    )
    from .models.ae.modeling_ae import AutoencoderModel
    from .models.base.modeling_base import BaseAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "AutoencoderModel",
            "AutoencoderDataset",
            "BaseAutoencoderModel",
            "CachedDataset",
            "DatasetLoaders",
            "DatasetSplits",
            "EmbeddingMatrix",
            "EmbeddingTensorDataset",
            "GloVeDataset",
            "PreTrainedAutoencoderModel",
            "create_dataloaders",
            "load_embedding_artifact",
            "load_text_embedding_matrix",
            "split_dataset",
        ]
    )
