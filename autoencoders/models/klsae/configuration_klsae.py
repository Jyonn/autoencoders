"""Configuration for KL sparse autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class KLSparseAutoencoderConfig(AutoencoderConfig):
    """Configuration for a KL-sparse autoencoder."""

    model_type = "kl_sparse_autoencoder"

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
        target_activation: float = 0.05,
        **kwargs,
    ) -> None:
        if sparsity_weight < 0:
            raise ValueError("sparsity_weight must be non-negative.")
        if not 0 < target_activation < 1:
            raise ValueError("target_activation must be between 0 and 1.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            sparsity_weight=sparsity_weight,
            target_activation=target_activation,
            **kwargs,
        )
