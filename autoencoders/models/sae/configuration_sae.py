"""Configuration for sparse autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class SparseAutoencoderConfig(AutoencoderConfig):
    """Configuration for a sparse autoencoder."""

    model_type = "sparse_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        sparsity_weight: float = 1e-3,
        **kwargs,
    ) -> None:
        if sparsity_weight < 0:
            raise ValueError("sparsity_weight must be non-negative.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            sparsity_weight=sparsity_weight,
            **kwargs,
        )
