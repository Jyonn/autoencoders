"""Configuration for MMD-VAE models."""

from __future__ import annotations

from ..infovae.configuration_infovae import InformationVariationalAutoencoderConfig


class MMDVariationalAutoencoderConfig(InformationVariationalAutoencoderConfig):
    """Configuration for an MMD-VAE."""

    model_type = "mmd_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        kl_weight: float = 0.0,
        mmd_weight: float = 10.0,
        mmd_bandwidths: list[float] | None = None,
        free_bits: float = 0.02,
        kl_warmup_epochs: int = 0,
        kl_start_weight: float = 0.0,
        use_mean_in_eval: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            mmd_weight=mmd_weight,
            mmd_bandwidths=mmd_bandwidths,
            free_bits=free_bits,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_start_weight=kl_start_weight,
            use_mean_in_eval=use_mean_in_eval,
            **kwargs,
        )
