"""Top-level package for the autoencoders library."""

from .configuration_utils import PretrainedConfig
from .modeling_outputs import AutoencoderExport, AutoencoderOutput
from .models.ae.configuration_ae import AutoencoderConfig
from .models.base.configuration_base import BaseAutoencoderConfig
from .models.betavae.configuration_betavae import BetaVariationalAutoencoderConfig
from .models.dae.configuration_dae import DenoisingAutoencoderConfig
from .models.sae.configuration_sae import SparseAutoencoderConfig
from .models.vae.configuration_vae import VariationalAutoencoderConfig

__all__ = [
    "AutoencoderConfig",
    "AutoencoderExport",
    "AutoencoderOutput",
    "BaseAutoencoderConfig",
    "BetaVariationalAutoencoderConfig",
    "DenoisingAutoencoderConfig",
    "PretrainedConfig",
    "SparseAutoencoderConfig",
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
    from .models.betavae.modeling_betavae import BetaVariationalAutoencoderModel
    from .models.dae.modeling_dae import DenoisingAutoencoderModel
    from .models.loading import load_model
    from .models.sae.modeling_sae import SparseAutoencoderModel
    from .models.vae.modeling_vae import VariationalAutoencoderModel
    from .training import (
        AutoencoderTrainer,
        TrainerDisplayConfig,
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
            "BetaVariationalAutoencoderModel",
            "CachedDataset",
            "DatasetLoaders",
            "DatasetSplits",
            "DenoisingAutoencoderModel",
            "EmbeddingMatrix",
            "EmbeddingTensorDataset",
            "GloVeDataset",
            "PreTrainedAutoencoderModel",
            "SparseAutoencoderModel",
            "TrainerDisplayConfig",
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
