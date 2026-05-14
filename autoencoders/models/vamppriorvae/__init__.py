"""VampPrior variational autoencoder models."""

from .configuration_vamppriorvae import VampPriorVariationalAutoencoderConfig

__all__ = ["VampPriorVariationalAutoencoderConfig"]

try:
    from .modeling_vamppriorvae import VampPriorVariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("VampPriorVariationalAutoencoderModel")
