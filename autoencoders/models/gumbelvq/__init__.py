"""Gumbel-VQ models."""

from .configuration_gumbelvq import GumbelQuantizedAutoencoderConfig

__all__ = ["GumbelQuantizedAutoencoderConfig"]

try:
    from .modeling_gumbelvq import GumbelQuantizedAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("GumbelQuantizedAutoencoderModel")
