"""Model loading helpers."""

from __future__ import annotations

from typing import Any

from .ae.configuration_ae import AutoencoderConfig
from .ae.modeling_ae import AutoencoderModel
from .dae.configuration_dae import DenoisingAutoencoderConfig
from .dae.modeling_dae import DenoisingAutoencoderModel


def load_model(name: str, **kwargs: Any):
    """Construct a named autoencoder model from config kwargs."""

    normalized_name = name.strip().lower()
    if normalized_name in {"ae", "autoencoder"}:
        return AutoencoderModel(AutoencoderConfig(**kwargs))
    if normalized_name in {"dae", "denoising_autoencoder", "denoising-autoencoder"}:
        return DenoisingAutoencoderModel(DenoisingAutoencoderConfig(**kwargs))
    raise ValueError(
        f"Unknown model {name!r}. Available models: 'ae', 'dae'."
    )

