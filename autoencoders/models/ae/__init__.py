"""Vanilla autoencoder models."""

from .configuration_ae import AutoencoderConfig

__all__ = ["AutoencoderConfig"]

try:
    from .modeling_ae import AutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("AutoencoderModel")
