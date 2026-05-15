"""Configuration for information variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class InformationVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for an InfoVAE with MMD prior matching."""

    model_type = "information_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        kl_weight: float = 0.1,
        mmd_weight: float = 5.0,
        mmd_bandwidths: list[float] | None = None,
        free_bits: float = 0.02,
        kl_warmup_epochs: int = 20,
        kl_start_weight: float = 0.0,
        use_mean_in_eval: bool = True,
        **kwargs,
    ) -> None:
        if mmd_weight < 0:
            raise ValueError("mmd_weight must be non-negative.")
        if mmd_bandwidths is None:
            mmd_bandwidths = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        if not mmd_bandwidths:
            raise ValueError("mmd_bandwidths must not be empty.")
        if any(bandwidth <= 0 for bandwidth in mmd_bandwidths):
            raise ValueError("mmd_bandwidths must contain only positive values.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_start_weight=kl_start_weight,
            use_mean_in_eval=use_mean_in_eval,
            mmd_weight=mmd_weight,
            mmd_bandwidths=list(mmd_bandwidths),
            **kwargs,
        )
