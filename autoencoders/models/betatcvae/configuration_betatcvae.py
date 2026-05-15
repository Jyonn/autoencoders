"""Configuration for Beta-TCVAE models."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class BetaTCVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a Beta-TCVAE."""

    model_type = "beta_tc_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        kl_weight: float = 1.0,
        free_bits: float = 0.02,
        kl_warmup_epochs: int = 20,
        kl_start_weight: float = 0.0,
        use_mean_in_eval: bool = True,
        mutual_information_weight: float = 1.0,
        total_correlation_weight: float = 6.0,
        dimension_wise_kl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if mutual_information_weight < 0:
            raise ValueError("mutual_information_weight must be non-negative.")
        if total_correlation_weight < 0:
            raise ValueError("total_correlation_weight must be non-negative.")
        if dimension_wise_kl_weight < 0:
            raise ValueError("dimension_wise_kl_weight must be non-negative.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_start_weight=kl_start_weight,
            use_mean_in_eval=use_mean_in_eval,
            mutual_information_weight=mutual_information_weight,
            total_correlation_weight=total_correlation_weight,
            dimension_wise_kl_weight=dimension_wise_kl_weight,
            **kwargs,
        )
