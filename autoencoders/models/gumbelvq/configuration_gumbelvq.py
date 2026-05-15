"""Configuration for Gumbel-VQ models."""

from __future__ import annotations

from ..base.configuration_vq import BaseVectorQuantizedAutoencoderConfig


class GumbelQuantizedAutoencoderConfig(BaseVectorQuantizedAutoencoderConfig):
    """Configuration for a Gumbel-softmax quantized autoencoder."""

    model_type = "gumbel_quantized_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        codebook_weight: float = 0.0,
        use_ema_codebook: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_reset: bool = False,
        dead_code_threshold: int = 0,
        temperature: float = 1.0,
        straight_through: bool = True,
        **kwargs,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            use_ema_codebook=use_ema_codebook,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
            dead_code_reset=dead_code_reset,
            dead_code_threshold=dead_code_threshold,
            temperature=temperature,
            straight_through=straight_through,
            **kwargs,
        )
