"""Beta-TCVAE models."""

from .configuration_betatcvae import BetaTCVariationalAutoencoderConfig

__all__ = ["BetaTCVariationalAutoencoderConfig"]

try:
    from .modeling_betatcvae import BetaTCVariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("BetaTCVariationalAutoencoderModel")
