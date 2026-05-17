"""PyTorch implementation of a basic feed-forward autoencoder."""

from ..base.modeling_base import BaseAutoencoderModel
from .configuration_ae import AutoencoderConfig


class AutoencoderModel(BaseAutoencoderModel):
    """A deterministic MLP autoencoder for vector-like feature inputs."""

    config_class = AutoencoderConfig
    config: AutoencoderConfig
