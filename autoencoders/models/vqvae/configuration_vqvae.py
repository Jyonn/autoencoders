"""Configuration for vector-quantized variational autoencoders."""

from __future__ import annotations

from ..base.configuration_vq import BaseVectorQuantizedAutoencoderConfig


class VectorQuantizedAutoencoderConfig(BaseVectorQuantizedAutoencoderConfig):
    """Configuration for a vector-quantized autoencoder."""

    model_type = "vector_quantized_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        use_ema_codebook: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            use_ema_codebook=use_ema_codebook,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
            **kwargs,
        )
