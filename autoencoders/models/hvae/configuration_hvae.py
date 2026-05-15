"""Configuration for hierarchical variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class HierarchicalVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a hierarchical variational autoencoder."""

    model_type = "hierarchical_variational_autoencoder"

    def __init__(
        self,
        top_latent_dim: int | None = None,
        **kwargs,
    ) -> None:
        latent_dim = kwargs.get("latent_dim")
        if latent_dim is None:
            raise TypeError("HierarchicalVariationalAutoencoderConfig requires `latent_dim`.")
        top_latent_dim = latent_dim if top_latent_dim is None else top_latent_dim
        if top_latent_dim <= 0:
            raise ValueError("top_latent_dim must be a positive integer.")
        self.top_latent_dim = top_latent_dim
        super().__init__(**kwargs)
