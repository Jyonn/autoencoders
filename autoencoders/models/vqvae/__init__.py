"""Vector-quantized variational autoencoder models."""

from .configuration_vqvae import VectorQuantizedAutoencoderConfig

__all__ = ["VectorQuantizedAutoencoderConfig"]

try:
    from .modeling_vqvae import VectorQuantizedAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("VectorQuantizedAutoencoderModel")
