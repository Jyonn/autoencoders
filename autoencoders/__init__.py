"""Top-level package for the autoencoders library."""

from .configuration_utils import PretrainedConfig
from .modeling_outputs import AutoencoderOutput
from .models.ae.configuration_ae import AutoencoderConfig
from .models.base.configuration_base import BaseAutoencoderConfig
from .models.dae.configuration_dae import DenoisingAutoencoderConfig
from .models.vae.configuration_vae import VariationalAutoencoderConfig

__all__ = [
    "AutoencoderConfig",
    "AutoencoderOutput",
    "BaseAutoencoderConfig",
    "DenoisingAutoencoderConfig",
    "PretrainedConfig",
    "VariationalAutoencoderConfig",
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
        load_dataset,
        load_embedding_artifact,
        load_text_embedding_matrix,
        split_dataset,
    )
    from .models.ae.modeling_ae import AutoencoderModel
    from .models.base.modeling_base import BaseAutoencoderModel
    from .models.dae.modeling_dae import DenoisingAutoencoderModel
    from .models.loading import load_model
    from .models.vae.modeling_vae import VariationalAutoencoderModel
    from .training import (
        AutoencoderTrainer,
        TrainingArguments,
        VAETrainer,
        VAETrainingArguments,
        resolve_device,
        set_seed,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "AutoencoderModel",
            "AutoencoderTrainer",
            "AutoencoderDataset",
            "BaseAutoencoderModel",
            "CachedDataset",
            "DatasetLoaders",
            "DatasetSplits",
            "DenoisingAutoencoderModel",
            "EmbeddingMatrix",
            "EmbeddingTensorDataset",
            "GloVeDataset",
            "PreTrainedAutoencoderModel",
            "TrainingArguments",
            "VAETrainer",
            "VAETrainingArguments",
            "VariationalAutoencoderModel",
            "create_dataloaders",
            "load_dataset",
            "load_model",
            "load_embedding_artifact",
            "load_text_embedding_matrix",
            "resolve_device",
            "set_seed",
            "split_dataset",
        ]
    )
