"""Configuration for residual-quantized autoencoders."""

from __future__ import annotations

from ..vqvae.configuration_vqvae import VectorQuantizedAutoencoderConfig


class ResidualQuantizedAutoencoderConfig(VectorQuantizedAutoencoderConfig):
    """Configuration for a residual-quantized autoencoder."""

    model_type = "residual_quantized_autoencoder"

    def __init__(
        self,
        num_quantizers: int = 2,
        **kwargs,
    ) -> None:
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be a positive integer.")
        if kwargs.get("use_ema_codebook"):
            raise ValueError(
                "ResidualQuantizedAutoencoderConfig does not support `use_ema_codebook=True`; "
                "reference RQ-VAE trains residual codebooks directly without EMA updates."
            )
        self.num_quantizers = num_quantizers
        super().__init__(**kwargs)
        self.validate_sinkhorn_slot_count(self.num_quantizers, "ResidualQuantizedAutoencoderConfig")
