"""Base configuration shared across autoencoder-family models."""

from __future__ import annotations

from ...configuration_utils import PretrainedConfig


class BaseAutoencoderConfig(PretrainedConfig):
    """Base config for latent models that reconstruct their inputs."""

    model_type = "base_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        reconstruction_loss: str = "mse",
        **kwargs,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if latent_dim is not None and latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if reconstruction_loss not in {"mse", "l1"}:
            raise ValueError("reconstruction_loss must be one of: 'mse', 'l1'.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss
        super().__init__(**kwargs)
