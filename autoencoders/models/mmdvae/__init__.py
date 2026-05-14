"""MMD-VAE models."""

from .configuration_mmdvae import MMDVariationalAutoencoderConfig

__all__ = ["MMDVariationalAutoencoderConfig"]

try:
    from .modeling_mmdvae import MMDVariationalAutoencoderModel
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("MMDVariationalAutoencoderModel")
