"""Configuration for the basic deterministic autoencoder."""

from __future__ import annotations

from ..base.configuration_base import BaseAutoencoderConfig


class AutoencoderConfig(BaseAutoencoderConfig):
    """Configuration for a feed-forward autoencoder."""

    model_type = "autoencoder"
