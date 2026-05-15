"""Configuration for DIP-VAE models."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class DIPVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a DIP-VAE."""

    model_type = "dip_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        kl_weight: float = 0.1,
        free_bits: float = 0.02,
        kl_warmup_epochs: int = 20,
        kl_start_weight: float = 0.0,
        use_mean_in_eval: bool = True,
        dip_weight: float = 10.0,
        dip_offdiag_weight: float = 1.0,
        dip_diag_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if dip_weight < 0:
            raise ValueError("dip_weight must be non-negative.")
        if dip_offdiag_weight < 0:
            raise ValueError("dip_offdiag_weight must be non-negative.")
        if dip_diag_weight < 0:
            raise ValueError("dip_diag_weight must be non-negative.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_start_weight=kl_start_weight,
            use_mean_in_eval=use_mean_in_eval,
            dip_weight=dip_weight,
            dip_offdiag_weight=dip_offdiag_weight,
            dip_diag_weight=dip_diag_weight,
            **kwargs,
        )
