"""Top-level package for the autoencoders library."""

from .models.aae.configuration_aae import AdversarialAutoencoderConfig
from .configuration_utils import PretrainedConfig
from .modeling_outputs import AutoencoderExport, AutoencoderOutput
from .models.ae.configuration_ae import AutoencoderConfig
from .models.cae.configuration_cae import ContractiveAutoencoderConfig
from .models.base.configuration_base import BaseAutoencoderConfig
from .models.base.configuration_vae import BaseVariationalAutoencoderConfig
from .models.base.configuration_vq import BaseVectorQuantizedAutoencoderConfig
from .models.betavae.configuration_betavae import BetaVariationalAutoencoderConfig
from .models.dae.configuration_dae import DenoisingAutoencoderConfig
from .models.dvae.configuration_dvae import DenoisingVariationalAutoencoderConfig
from .models.fsq.configuration_fsq import FiniteScalarQuantizedAutoencoderConfig
from .models.hvae.configuration_hvae import HierarchicalVariationalAutoencoderConfig
from .models.klsae.configuration_klsae import KLSparseAutoencoderConfig
from .models.pqvae.configuration_pqvae import ProductQuantizedAutoencoderConfig
from .models.rqvae.configuration_rqvae import ResidualQuantizedAutoencoderConfig
from .models.sae.configuration_sae import SparseAutoencoderConfig
from .models.topksae.configuration_topksae import TopKSparseAutoencoderConfig
from .models.vae.configuration_vae import VariationalAutoencoderConfig
from .models.wae.configuration_wae import WassersteinAutoencoderConfig
from .models.vqvae.configuration_vqvae import VectorQuantizedAutoencoderConfig

__all__ = [
    "AdversarialAutoencoderConfig",
    "AutoencoderConfig",
    "AutoencoderExport",
    "AutoencoderOutput",
    "BaseAutoencoderConfig",
    "BaseVariationalAutoencoderConfig",
    "BaseVectorQuantizedAutoencoderConfig",
    "BetaVariationalAutoencoderConfig",
    "ContractiveAutoencoderConfig",
    "DenoisingAutoencoderConfig",
    "DenoisingVariationalAutoencoderConfig",
    "FiniteScalarQuantizedAutoencoderConfig",
    "HierarchicalVariationalAutoencoderConfig",
    "KLSparseAutoencoderConfig",
    "PretrainedConfig",
    "ProductQuantizedAutoencoderConfig",
    "ResidualQuantizedAutoencoderConfig",
    "SparseAutoencoderConfig",
    "TopKSparseAutoencoderConfig",
    "VariationalAutoencoderConfig",
    "WassersteinAutoencoderConfig",
    "VectorQuantizedAutoencoderConfig",
]

try:
    from .models.aae.modeling_aae import AdversarialAutoencoderModel
    from .modeling_utils import PreTrainedAutoencoderModel
    from .data import (
        AutoencoderDataset,
        CachedDataset,
        ConceptNetNumberbatchDataset,
        DatasetLoaders,
        DatasetSplits,
        EmbeddingMatrix,
        EmbeddingTensorDataset,
        FastTextEnglishDataset,
        GloVeDataset,
        create_dataloaders,
        load_dataset,
        load_embedding_artifact,
        load_text_embedding_matrix,
        split_dataset,
    )
    from .models.ae.modeling_ae import AutoencoderModel
    from .models.base.modeling_base import BaseAutoencoderModel
    from .models.base.modeling_vae import BaseVariationalAutoencoderModel
    from .models.base.modeling_vq import BaseVectorQuantizedAutoencoderModel
    from .models.betavae.modeling_betavae import BetaVariationalAutoencoderModel
    from .models.cae.modeling_cae import ContractiveAutoencoderModel
    from .models.dae.modeling_dae import DenoisingAutoencoderModel
    from .models.dvae.modeling_dvae import DenoisingVariationalAutoencoderModel
    from .models.fsq.modeling_fsq import FiniteScalarQuantizedAutoencoderModel
    from .models.hvae.modeling_hvae import HierarchicalVariationalAutoencoderModel
    from .models.klsae.modeling_klsae import KLSparseAutoencoderModel
    from .models.loading import load_model
    from .models.pqvae.modeling_pqvae import ProductQuantizedAutoencoderModel
    from .models.rqvae.modeling_rqvae import ResidualQuantizedAutoencoderModel
    from .models.sae.modeling_sae import SparseAutoencoderModel
    from .models.topksae.modeling_topksae import TopKSparseAutoencoderModel
    from .models.vae.modeling_vae import VariationalAutoencoderModel
    from .models.wae.modeling_wae import WassersteinAutoencoderModel
    from .models.vqvae.modeling_vqvae import VectorQuantizedAutoencoderModel
    from .training import (
        AdversarialAutoencoderTrainer,
        AdversarialAutoencoderTrainingArguments,
        AutoencoderTrainer,
        ContractiveAutoencoderTrainer,
        QuantizedAutoencoderTrainer,
        QuantizedAutoencoderTrainingArguments,
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
            "AdversarialAutoencoderModel",
            "AdversarialAutoencoderTrainer",
            "AdversarialAutoencoderTrainingArguments",
            "AutoencoderModel",
            "AutoencoderTrainer",
            "AutoencoderDataset",
            "BaseAutoencoderModel",
            "BaseVariationalAutoencoderModel",
            "BaseVectorQuantizedAutoencoderModel",
            "BetaVariationalAutoencoderModel",
            "CachedDataset",
            "ContractiveAutoencoderModel",
            "ContractiveAutoencoderTrainer",
            "ConceptNetNumberbatchDataset",
            "DatasetLoaders",
            "DatasetSplits",
            "DenoisingAutoencoderModel",
            "DenoisingVariationalAutoencoderModel",
            "EmbeddingMatrix",
            "EmbeddingTensorDataset",
            "FastTextEnglishDataset",
            "FiniteScalarQuantizedAutoencoderModel",
            "GloVeDataset",
            "HierarchicalVariationalAutoencoderModel",
            "KLSparseAutoencoderModel",
            "PreTrainedAutoencoderModel",
            "ProductQuantizedAutoencoderModel",
            "QuantizedAutoencoderTrainer",
            "QuantizedAutoencoderTrainingArguments",
            "ResidualQuantizedAutoencoderModel",
            "SparseAutoencoderModel",
            "TopKSparseAutoencoderModel",
            "TrainerDisplayConfig",
            "TrainingArguments",
            "VAETrainer",
            "VAETrainingArguments",
            "VariationalAutoencoderModel",
            "VectorQuantizedAutoencoderModel",
            "WassersteinAutoencoderModel",
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
