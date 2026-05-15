"""Configuration for FactorVAE models."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class FactorVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a FactorVAE."""

    model_type = "factor_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        kl_weight: float = 0.1,
        free_bits: float = 0.02,
        kl_warmup_epochs: int = 20,
        kl_start_weight: float = 0.0,
        use_mean_in_eval: bool = True,
        tc_weight: float = 10.0,
        discriminator_hidden_dims: list[int] | None = None,
        **kwargs,
    ) -> None:
        if tc_weight < 0:
            raise ValueError("tc_weight must be non-negative.")
        if discriminator_hidden_dims is None:
            discriminator_hidden_dims = [128, 64]
        if any(dim <= 0 for dim in discriminator_hidden_dims):
            raise ValueError("discriminator_hidden_dims must contain positive integers.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_start_weight=kl_start_weight,
            use_mean_in_eval=use_mean_in_eval,
            tc_weight=tc_weight,
            discriminator_hidden_dims=list(discriminator_hidden_dims),
            **kwargs,
        )
