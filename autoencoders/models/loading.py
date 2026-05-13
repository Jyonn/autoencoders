"""Model loading helpers."""

from __future__ import annotations

from typing import Any

from .ae.configuration_ae import AutoencoderConfig
from .ae.modeling_ae import AutoencoderModel
from .betavae.configuration_betavae import BetaVariationalAutoencoderConfig
from .betavae.modeling_betavae import BetaVariationalAutoencoderModel
from .dae.configuration_dae import DenoisingAutoencoderConfig
from .dae.modeling_dae import DenoisingAutoencoderModel
from .sae.configuration_sae import SparseAutoencoderConfig
from .sae.modeling_sae import SparseAutoencoderModel
from .vae.configuration_vae import VariationalAutoencoderConfig
from .vae.modeling_vae import VariationalAutoencoderModel


def load_model(name: str, **kwargs: Any):
    """Construct a named autoencoder model from config kwargs."""

    if name == "ae":
        return AutoencoderModel(AutoencoderConfig(**kwargs))
    if name == "dae":
        return DenoisingAutoencoderModel(DenoisingAutoencoderConfig(**kwargs))
    if name == "sae":
        return SparseAutoencoderModel(SparseAutoencoderConfig(**kwargs))
    if name == "vae":
        return VariationalAutoencoderModel(VariationalAutoencoderConfig(**kwargs))
    if name == "betavae":
        return BetaVariationalAutoencoderModel(BetaVariationalAutoencoderConfig(**kwargs))
    raise ValueError(
        f"Unknown model {name!r}. Available models: 'ae', 'dae', 'sae', 'vae', 'betavae'."
    )
