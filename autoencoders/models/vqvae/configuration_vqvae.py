"""Configuration for vector-quantized variational autoencoders."""

from __future__ import annotations

from ..base.configuration_vq import BaseVectorQuantizedAutoencoderConfig


class VectorQuantizedAutoencoderConfig(BaseVectorQuantizedAutoencoderConfig):
    """Configuration for a vector-quantized autoencoder."""

    model_type = "vector_quantized_autoencoder"
