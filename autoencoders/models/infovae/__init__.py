"""Information variational autoencoder models."""

from .configuration_infovae import InformationVariationalAutoencoderConfig

__all__ = ["InformationVariationalAutoencoderConfig"]

try:
    from .modeling_infovae import InformationVariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("InformationVariationalAutoencoderModel")
