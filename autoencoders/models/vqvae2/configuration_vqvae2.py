"""Configuration for hierarchical vector-quantized autoencoders."""

from __future__ import annotations

from ..base.configuration_vq import BaseVectorQuantizedAutoencoderConfig


class HierarchicalVectorQuantizedAutoencoderConfig(BaseVectorQuantizedAutoencoderConfig):
    """Configuration for a two-level VQ-VAE-2 style autoencoder."""

    model_type = "hierarchical_vector_quantized_autoencoder"

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
        dead_code_reset: bool = False,
        dead_code_threshold: int = 0,
        top_latent_dim: int | None = None,
        **kwargs,
    ) -> None:
        if top_latent_dim is None:
            top_latent_dim = latent_dim
        if top_latent_dim <= 0:
            raise ValueError("top_latent_dim must be positive.")
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
            dead_code_reset=dead_code_reset,
            dead_code_threshold=dead_code_threshold,
            top_latent_dim=top_latent_dim,
            **kwargs,
        )
