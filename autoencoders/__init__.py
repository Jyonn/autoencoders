"""Top-level package for the autoencoders library."""

from .models.aae.configuration_aae import AdversarialAutoencoderConfig
from .configuration_utils import PretrainedConfig
from .modules import (
    BaseAutoencoderModule,
    BaseAutoencoderModuleConfig,
    MLPModule,
    MLPModuleConfig,
    build_mlp_backbone_kwargs,
    build_mlp_backbone_kwargs_from_model_config,
    get_module_class,
    get_module_modules,
)
from .modeling_outputs import (
    AdversarialAutoencoderOutput,
    AutoencoderExport,
    AutoencoderOutput,
    BaseAutoencoderOutput,
    ContractiveAutoencoderOutput,
    DenoisingAutoencoderOutput,
    DenoisingVariationalAutoencoderOutput,
    DIPVariationalAutoencoderOutput,
    FactorVariationalAutoencoderOutput,
    FiniteScalarQuantizedAutoencoderOutput,
    GumbelQuantizedAutoencoderOutput,
    HierarchicalVariationalAutoencoderOutput,
    HierarchicalQuantizedAutoencoderOutput,
    InformationVariationalAutoencoderOutput,
    KLSparseAutoencoderOutput,
    MMDVariationalAutoencoderOutput,
    QuantizedAutoencoderOutput,
    ResidualFiniteScalarQuantizedAutoencoderOutput,
    SparseAutoencoderOutput,
    BetaTCVariationalAutoencoderOutput,
    TopKSparseAutoencoderOutput,
    VampPriorVariationalAutoencoderOutput,
    VariationalAutoencoderOutput,
    WassersteinAutoencoderOutput,
)
from .models.ae.configuration_ae import AutoencoderConfig
from .models.cae.configuration_cae import ContractiveAutoencoderConfig
from .models.base.configuration_base import BaseAutoencoderConfig
from .models.base.configuration_vae import BaseVariationalAutoencoderConfig
from .models.base.configuration_vq import BaseVectorQuantizedAutoencoderConfig
from .models.betavae.configuration_betavae import BetaVariationalAutoencoderConfig
from .models.betatcvae.configuration_betatcvae import BetaTCVariationalAutoencoderConfig
from .models.dae.configuration_dae import DenoisingAutoencoderConfig
from .models.dipvae.configuration_dipvae import DIPVariationalAutoencoderConfig
from .models.dvae.configuration_dvae import DenoisingVariationalAutoencoderConfig
from .models.factorvae.configuration_factorvae import FactorVariationalAutoencoderConfig
from .models.fsq.configuration_fsq import FiniteScalarQuantizedAutoencoderConfig
from .models.gumbelvq.configuration_gumbelvq import GumbelQuantizedAutoencoderConfig
from .models.hvae.configuration_hvae import HierarchicalVariationalAutoencoderConfig
from .models.infovae.configuration_infovae import InformationVariationalAutoencoderConfig
from .models.klsae.configuration_klsae import KLSparseAutoencoderConfig
from .models.mmdvae.configuration_mmdvae import MMDVariationalAutoencoderConfig
from .models.pqvae.configuration_pqvae import ProductQuantizedAutoencoderConfig
from .models.rfsq.configuration_rfsq import ResidualFiniteScalarQuantizedAutoencoderConfig
from .models.rqvae.configuration_rqvae import ResidualQuantizedAutoencoderConfig
from .models.sae.configuration_sae import SparseAutoencoderConfig
from .models.topksae.configuration_topksae import TopKSparseAutoencoderConfig
from .models.vae.configuration_vae import VariationalAutoencoderConfig
from .models.vamppriorvae.configuration_vamppriorvae import VampPriorVariationalAutoencoderConfig
from .models.vqvae2.configuration_vqvae2 import HierarchicalVectorQuantizedAutoencoderConfig
from .models.wae.configuration_wae import WassersteinAutoencoderConfig
from .models.vqvae.configuration_vqvae import VectorQuantizedAutoencoderConfig

__all__ = [
    "AdversarialAutoencoderConfig",
    "AdversarialAutoencoderOutput",
    "AutoencoderConfig",
    "AutoencoderExport",
    "AutoencoderOutput",
    "BaseAutoencoderOutput",
    "BaseAutoencoderConfig",
    "BaseAutoencoderModule",
    "BaseAutoencoderModuleConfig",
    "build_mlp_backbone_kwargs",
    "build_mlp_backbone_kwargs_from_model_config",
    "BaseVariationalAutoencoderConfig",
    "BaseVectorQuantizedAutoencoderConfig",
    "BetaVariationalAutoencoderConfig",
    "BetaTCVariationalAutoencoderConfig",
    "BetaTCVariationalAutoencoderOutput",
    "ContractiveAutoencoderConfig",
    "ContractiveAutoencoderOutput",
    "DenoisingAutoencoderConfig",
    "DenoisingAutoencoderOutput",
    "DenoisingVariationalAutoencoderConfig",
    "DenoisingVariationalAutoencoderOutput",
    "DIPVariationalAutoencoderConfig",
    "DIPVariationalAutoencoderOutput",
    "FactorVariationalAutoencoderConfig",
    "FactorVariationalAutoencoderOutput",
    "FiniteScalarQuantizedAutoencoderConfig",
    "FiniteScalarQuantizedAutoencoderOutput",
    "GumbelQuantizedAutoencoderConfig",
    "GumbelQuantizedAutoencoderOutput",
    "HierarchicalQuantizedAutoencoderOutput",
    "HierarchicalVectorQuantizedAutoencoderConfig",
    "HierarchicalVariationalAutoencoderConfig",
    "HierarchicalVariationalAutoencoderOutput",
    "InformationVariationalAutoencoderConfig",
    "InformationVariationalAutoencoderOutput",
    "KLSparseAutoencoderConfig",
    "KLSparseAutoencoderOutput",
    "MMDVariationalAutoencoderConfig",
    "MMDVariationalAutoencoderOutput",
    "MLPModule",
    "MLPModuleConfig",
    "PretrainedConfig",
    "ProductQuantizedAutoencoderConfig",
    "QuantizedAutoencoderOutput",
    "ResidualFiniteScalarQuantizedAutoencoderConfig",
    "ResidualFiniteScalarQuantizedAutoencoderOutput",
    "ResidualQuantizedAutoencoderConfig",
    "SparseAutoencoderConfig",
    "SparseAutoencoderOutput",
    "TopKSparseAutoencoderConfig",
    "TopKSparseAutoencoderOutput",
    "VampPriorVariationalAutoencoderConfig",
    "VampPriorVariationalAutoencoderOutput",
    "VariationalAutoencoderConfig",
    "VariationalAutoencoderOutput",
    "WassersteinAutoencoderConfig",
    "WassersteinAutoencoderOutput",
    "VectorQuantizedAutoencoderConfig",
    "get_module_class",
    "get_module_modules",
]

try:
    from .models.aae.modeling_aae import AdversarialAutoencoderModel
    from .modeling_utils import PreTrainedAutoencoderModel
    from .data import (
        AutoencoderDataset,
        CachedDataset,
        CLIPBackedDataset,
        CLIPEmbeddingEncoder,
        CLIPRecord,
        ConceptNetNumberbatchDataset,
        DatasetLoaders,
        DatasetSplits,
        EmbeddingMatrix,
        EmbeddingTensorDataset,
        FastTextEnglishDataset,
        Flickr30kDataset,
        GloVeDataset,
        OpenCLIPEmbeddingEncoder,
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
    from .models.betatcvae.modeling_betatcvae import BetaTCVariationalAutoencoderModel
    from .models.cae.modeling_cae import ContractiveAutoencoderModel
    from .models.dae.modeling_dae import DenoisingAutoencoderModel
    from .models.dipvae.modeling_dipvae import DIPVariationalAutoencoderModel
    from .models.dvae.modeling_dvae import DenoisingVariationalAutoencoderModel
    from .models.factorvae.modeling_factorvae import FactorVariationalAutoencoderModel
    from .models.fsq.modeling_fsq import FiniteScalarQuantizedAutoencoderModel
    from .models.gumbelvq.modeling_gumbelvq import GumbelQuantizedAutoencoderModel
    from .models.hvae.modeling_hvae import HierarchicalVariationalAutoencoderModel
    from .models.infovae.modeling_infovae import InformationVariationalAutoencoderModel
    from .models.klsae.modeling_klsae import KLSparseAutoencoderModel
    from .models.loading import load_model
    from .models.mmdvae.modeling_mmdvae import MMDVariationalAutoencoderModel
    from .models.pqvae.modeling_pqvae import ProductQuantizedAutoencoderModel
    from .models.rfsq.modeling_rfsq import ResidualFiniteScalarQuantizedAutoencoderModel
    from .models.rqvae.modeling_rqvae import ResidualQuantizedAutoencoderModel
    from .models.sae.modeling_sae import SparseAutoencoderModel
    from .models.topksae.modeling_topksae import TopKSparseAutoencoderModel
    from .models.vae.modeling_vae import VariationalAutoencoderModel
    from .models.vamppriorvae.modeling_vamppriorvae import VampPriorVariationalAutoencoderModel
    from .models.wae.modeling_wae import WassersteinAutoencoderModel
    from .models.vqvae2.modeling_vqvae2 import HierarchicalVectorQuantizedAutoencoderModel
    from .models.vqvae.modeling_vqvae import VectorQuantizedAutoencoderModel
    from .training import (
        AETrainer,
        AdversarialAutoencoderTrainer,
        AdversarialAutoencoderTrainingArguments,
        FactorVAETrainer,
        FactorVariationalAutoencoderTrainingArguments,
        TrainerDisplay,
        TrainerDisplayConfig,
        TrainingArguments,
        VAETrainer,
        VQTrainer,
        resolve_device,
        set_seed,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "AETrainer",
            "AdversarialAutoencoderModel",
            "AdversarialAutoencoderTrainer",
            "AdversarialAutoencoderTrainingArguments",
            "AutoencoderModel",
            "AutoencoderDataset",
            "BaseAutoencoderModel",
            "BaseVariationalAutoencoderModel",
            "BaseVectorQuantizedAutoencoderModel",
            "BetaVariationalAutoencoderModel",
            "BetaTCVariationalAutoencoderModel",
            "CachedDataset",
            "CLIPBackedDataset",
            "CLIPEmbeddingEncoder",
            "CLIPRecord",
            "ContractiveAutoencoderModel",
            "ConceptNetNumberbatchDataset",
            "DatasetLoaders",
            "DatasetSplits",
            "DenoisingAutoencoderModel",
            "DIPVariationalAutoencoderModel",
            "DenoisingVariationalAutoencoderModel",
            "EmbeddingMatrix",
            "EmbeddingTensorDataset",
            "FactorVAETrainer",
            "FactorVariationalAutoencoderModel",
            "FactorVariationalAutoencoderTrainingArguments",
            "FastTextEnglishDataset",
            "FiniteScalarQuantizedAutoencoderModel",
            "Flickr30kDataset",
            "GloVeDataset",
            "GumbelQuantizedAutoencoderModel",
            "HierarchicalVectorQuantizedAutoencoderModel",
            "HierarchicalVariationalAutoencoderModel",
            "InformationVariationalAutoencoderModel",
            "KLSparseAutoencoderModel",
            "MMDVariationalAutoencoderModel",
            "OpenCLIPEmbeddingEncoder",
            "PreTrainedAutoencoderModel",
            "ProductQuantizedAutoencoderModel",
            "ResidualFiniteScalarQuantizedAutoencoderModel",
            "ResidualQuantizedAutoencoderModel",
            "SparseAutoencoderModel",
            "TopKSparseAutoencoderModel",
            "TrainerDisplay",
            "TrainerDisplayConfig",
            "TrainingArguments",
            "VAETrainer",
            "VampPriorVariationalAutoencoderModel",
            "VariationalAutoencoderModel",
            "VQTrainer",
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
