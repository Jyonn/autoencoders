"""Beta variational autoencoder models."""

from .configuration_betavae import BetaVariationalAutoencoderConfig

__all__ = ["BetaVariationalAutoencoderConfig"]

try:
    from .modeling_betavae import BetaVariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("BetaVariationalAutoencoderModel")
