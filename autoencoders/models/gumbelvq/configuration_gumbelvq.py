"""Configuration for Gumbel-VQ models."""

from __future__ import annotations

from ..base.configuration_vq import BaseVectorQuantizedAutoencoderConfig


class GumbelQuantizedAutoencoderConfig(BaseVectorQuantizedAutoencoderConfig):
    """Configuration for a Gumbel-softmax quantized autoencoder."""

    model_type = "gumbel_quantized_autoencoder"

    def __init__(
        self,
        temperature: float = 1.0,
        straight_through: bool = True,
        **kwargs,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        kwargs.setdefault("codebook_weight", 0.0)
        self.temperature = temperature
        self.straight_through = straight_through
        super().__init__(**kwargs)
