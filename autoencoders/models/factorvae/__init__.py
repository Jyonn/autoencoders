"""FactorVAE models."""

from .configuration_factorvae import FactorVariationalAutoencoderConfig

__all__ = ["FactorVariationalAutoencoderConfig"]

try:
    from .modeling_factorvae import FactorVariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("FactorVariationalAutoencoderModel")
