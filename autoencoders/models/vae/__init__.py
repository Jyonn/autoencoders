"""Variational autoencoder models."""

from .configuration_vae import VariationalAutoencoderConfig

__all__ = ["VariationalAutoencoderConfig"]

try:
    from .modeling_vae import VariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("VariationalAutoencoderModel")

