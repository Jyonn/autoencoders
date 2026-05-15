"""Configuration for beta variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class BetaVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a beta-VAE."""

    model_type = "beta_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        beta: float = 4.0,
        free_bits: float = 0.02,
        use_mean_in_eval: bool = True,
        **kwargs,
    ) -> None:
        kl_weight = kwargs.pop("kl_weight", beta)

        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            use_mean_in_eval=use_mean_in_eval,
            beta=kl_weight,
            **kwargs,
        )
