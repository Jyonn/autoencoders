"""Configuration for MMD-VAE models."""

from __future__ import annotations

from ..infovae.configuration_infovae import InformationVariationalAutoencoderConfig


class MMDVariationalAutoencoderConfig(InformationVariationalAutoencoderConfig):
    """Configuration for an MMD-VAE."""

    model_type = "mmd_variational_autoencoder"

    def __init__(
        self,
        mmd_weight: float = 10.0,
        mmd_bandwidths: list[float] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("kl_weight", 0.0)
        kwargs.setdefault("mmd_weight", mmd_weight)
        kwargs.setdefault("mmd_bandwidths", mmd_bandwidths)
        kwargs.setdefault("free_bits", 0.02)
        kwargs.setdefault("kl_warmup_epochs", 0)
        kwargs.setdefault("kl_start_weight", 0.0)
        kwargs.setdefault("use_mean_in_eval", True)
        super().__init__(**kwargs)
