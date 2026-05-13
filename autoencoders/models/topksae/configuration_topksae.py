"""Configuration for top-k sparse autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class TopKSparseAutoencoderConfig(AutoencoderConfig):
    """Configuration for a top-k sparse autoencoder."""

    model_type = "topk_sparse_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        topk: int = 4,
        **kwargs,
    ) -> None:
        if topk <= 0:
            raise ValueError("topk must be a positive integer.")
        if topk > latent_dim:
            raise ValueError("topk must be less than or equal to latent_dim.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            topk=topk,
            **kwargs,
        )
