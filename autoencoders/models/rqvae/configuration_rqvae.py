"""Configuration for residual-quantized autoencoders."""

from __future__ import annotations

from ..vqvae.configuration_vqvae import VectorQuantizedAutoencoderConfig


class ResidualQuantizedAutoencoderConfig(VectorQuantizedAutoencoderConfig):
    """Configuration for a residual-quantized autoencoder."""

    model_type = "residual_quantized_autoencoder"

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
        use_ema_codebook: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        num_quantizers: int = 2,
        **kwargs,
    ) -> None:
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be a positive integer.")

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
            use_ema_codebook=use_ema_codebook,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
            num_quantizers=num_quantizers,
            **kwargs,
        )
