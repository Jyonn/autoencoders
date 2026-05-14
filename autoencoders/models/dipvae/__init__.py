"""DIP-VAE models."""

from .configuration_dipvae import DIPVariationalAutoencoderConfig

__all__ = ["DIPVariationalAutoencoderConfig"]

try:
    from .modeling_dipvae import DIPVariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("DIPVariationalAutoencoderModel")
