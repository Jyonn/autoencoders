"""Sparse autoencoder models."""

from .configuration_sae import SparseAutoencoderConfig

__all__ = ["SparseAutoencoderConfig"]

try:
    from .modeling_sae import SparseAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("SparseAutoencoderModel")
