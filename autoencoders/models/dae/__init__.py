"""Denoising autoencoder models."""

from .configuration_dae import DenoisingAutoencoderConfig

__all__ = ["DenoisingAutoencoderConfig"]

try:
    from .modeling_dae import DenoisingAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("DenoisingAutoencoderModel")

