"""Base configuration shared by vector-quantized autoencoder families."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class BaseVectorQuantizedAutoencoderConfig(AutoencoderConfig):
    """Base config for VQ-style autoencoders."""

    model_type = "base_vector_quantized_autoencoder"

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
        dead_code_reset: bool = False,
        dead_code_threshold: int = 0,
        **kwargs,
    ) -> None:
        if codebook_size <= 0:
            raise ValueError("codebook_size must be a positive integer.")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative.")
        if codebook_weight < 0:
            raise ValueError("codebook_weight must be non-negative.")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in the range [0, 1).")
        if ema_epsilon <= 0:
            raise ValueError("ema_epsilon must be positive.")
        if dead_code_threshold < 0:
            raise ValueError("dead_code_threshold must be non-negative.")

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
            dead_code_reset=dead_code_reset,
            dead_code_threshold=dead_code_threshold,
            **kwargs,
        )
