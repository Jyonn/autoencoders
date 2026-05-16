"""PyTorch implementation of a beta variational autoencoder."""

from __future__ import annotations

from ..vae.modeling_vae import VariationalAutoencoderModel
from .configuration_betavae import BetaVariationalAutoencoderConfig


class BetaVariationalAutoencoderModel(VariationalAutoencoderModel):
    """A beta-VAE that reuses the standard variational autoencoder implementation."""

    config_class = BetaVariationalAutoencoderConfig
    config: BetaVariationalAutoencoderConfig
