"""Configuration for vector-quantized variational autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class VectorQuantizedAutoencoderConfig(AutoencoderConfig):
    """Configuration for a vector-quantized autoencoder."""

    model_type = "vector_quantized_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if codebook_size <= 0:
            raise ValueError("codebook_size must be a positive integer.")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative.")
        if codebook_weight < 0:
            raise ValueError("codebook_weight must be non-negative.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            **kwargs,
        )
