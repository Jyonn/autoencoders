"""Residual finite-scalar quantized autoencoder models."""

from .configuration_rfsq import ResidualFiniteScalarQuantizedAutoencoderConfig

__all__ = ["ResidualFiniteScalarQuantizedAutoencoderConfig"]

try:
    from .modeling_rfsq import ResidualFiniteScalarQuantizedAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("ResidualFiniteScalarQuantizedAutoencoderModel")
