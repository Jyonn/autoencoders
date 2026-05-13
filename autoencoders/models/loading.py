"""Model loading helpers."""

from __future__ import annotations

from typing import Any

from .aae.configuration_aae import AdversarialAutoencoderConfig
from .aae.modeling_aae import AdversarialAutoencoderModel
from .ae.configuration_ae import AutoencoderConfig
from .ae.modeling_ae import AutoencoderModel
from .betavae.configuration_betavae import BetaVariationalAutoencoderConfig
from .betavae.modeling_betavae import BetaVariationalAutoencoderModel
from .cae.configuration_cae import ContractiveAutoencoderConfig
from .cae.modeling_cae import ContractiveAutoencoderModel
from .dae.configuration_dae import DenoisingAutoencoderConfig
from .dae.modeling_dae import DenoisingAutoencoderModel
from .pqvae.configuration_pqvae import ProductQuantizedAutoencoderConfig
from .pqvae.modeling_pqvae import ProductQuantizedAutoencoderModel
from .rqvae.configuration_rqvae import ResidualQuantizedAutoencoderConfig
from .rqvae.modeling_rqvae import ResidualQuantizedAutoencoderModel
from .sae.configuration_sae import SparseAutoencoderConfig
from .sae.modeling_sae import SparseAutoencoderModel
from .vae.configuration_vae import VariationalAutoencoderConfig
from .vae.modeling_vae import VariationalAutoencoderModel
from .wae.configuration_wae import WassersteinAutoencoderConfig
from .wae.modeling_wae import WassersteinAutoencoderModel
from .vqvae.configuration_vqvae import VectorQuantizedAutoencoderConfig
from .vqvae.modeling_vqvae import VectorQuantizedAutoencoderModel


def load_model(name: str, **kwargs: Any):
    """Construct a named autoencoder model from config kwargs."""

    if name == "ae":
        return AutoencoderModel(AutoencoderConfig(**kwargs))
    if name == "dae":
        return DenoisingAutoencoderModel(DenoisingAutoencoderConfig(**kwargs))
    if name == "cae":
        return ContractiveAutoencoderModel(ContractiveAutoencoderConfig(**kwargs))
    if name == "sae":
        return SparseAutoencoderModel(SparseAutoencoderConfig(**kwargs))
    if name == "vae":
        return VariationalAutoencoderModel(VariationalAutoencoderConfig(**kwargs))
    if name == "betavae":
        return BetaVariationalAutoencoderModel(BetaVariationalAutoencoderConfig(**kwargs))
    if name == "wae":
        return WassersteinAutoencoderModel(WassersteinAutoencoderConfig(**kwargs))
    if name == "aae":
        return AdversarialAutoencoderModel(AdversarialAutoencoderConfig(**kwargs))
    if name == "vqvae":
        return VectorQuantizedAutoencoderModel(VectorQuantizedAutoencoderConfig(**kwargs))
    if name == "pqvae":
        return ProductQuantizedAutoencoderModel(ProductQuantizedAutoencoderConfig(**kwargs))
    if name == "rqvae":
        return ResidualQuantizedAutoencoderModel(ResidualQuantizedAutoencoderConfig(**kwargs))
    raise ValueError(
        f"Unknown model {name!r}. Available models: 'ae', 'dae', 'cae', 'sae', 'vae', 'betavae', 'wae', 'aae', 'vqvae', 'pqvae', 'rqvae'."
    )
