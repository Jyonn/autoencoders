"""Base model abstractions."""

from .configuration_base import BaseAutoencoderConfig

__all__ = ["BaseAutoencoderConfig"]

try:
    from .modeling_base import BaseAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("BaseAutoencoderModel")
