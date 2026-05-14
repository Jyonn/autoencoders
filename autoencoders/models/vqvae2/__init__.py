"""VQ-VAE-2 models."""

from .configuration_vqvae2 import HierarchicalVectorQuantizedAutoencoderConfig

__all__ = ["HierarchicalVectorQuantizedAutoencoderConfig"]

try:
    from .modeling_vqvae2 import HierarchicalVectorQuantizedAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("HierarchicalVectorQuantizedAutoencoderModel")
