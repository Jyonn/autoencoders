"""Configuration for variational autoencoders."""

from __future__ import annotations

from ..base.configuration_vae import BaseVariationalAutoencoderConfig


class VariationalAutoencoderConfig(BaseVariationalAutoencoderConfig):
    """Configuration for a variational autoencoder."""

    model_type = "variational_autoencoder"
