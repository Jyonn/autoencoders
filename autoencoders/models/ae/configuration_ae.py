"""Configuration for the basic deterministic autoencoder."""

from __future__ import annotations

from ..base.configuration_base import BaseAutoencoderConfig


class AutoencoderConfig(BaseAutoencoderConfig):
    """Configuration for a feed-forward autoencoder."""

    model_type = "autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        **kwargs,
    ) -> None:

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            **kwargs,
        )
