"""Configuration for variational autoencoders."""

from __future__ import annotations

from ..base.configuration_vae import BaseVariationalAutoencoderConfig


class VariationalAutoencoderConfig(BaseVariationalAutoencoderConfig):
    """Configuration for a variational autoencoder."""

    model_type = "variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        kl_weight: float = 1.0,
        free_bits: float = 0.02,
        use_mean_in_eval: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            use_mean_in_eval=use_mean_in_eval,
            **kwargs,
        )
