"""Configuration for Wasserstein autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class WassersteinAutoencoderConfig(AutoencoderConfig):
    """Configuration for a Wasserstein autoencoder with MMD regularization."""

    model_type = "wasserstein_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        mmd_weight: float = 10.0,
        mmd_bandwidths: list[float] | None = None,
        **kwargs,
    ) -> None:
        if mmd_weight < 0:
            raise ValueError("mmd_weight must be non-negative.")
        if mmd_bandwidths is None:
            mmd_bandwidths = [0.1, 0.2, 0.5, 1.0, 2.0]
        if any(bandwidth <= 0 for bandwidth in mmd_bandwidths):
            raise ValueError("mmd_bandwidths must contain positive floats.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            mmd_weight=mmd_weight,
            mmd_bandwidths=list(mmd_bandwidths),
            **kwargs,
        )
