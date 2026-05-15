"""Configuration for information variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class InformationVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for an InfoVAE with MMD prior matching."""

    model_type = "information_variational_autoencoder"

    def __init__(
        self,
        mmd_weight: float = 5.0,
        mmd_bandwidths: list[float] | None = None,
        **kwargs,
    ) -> None:
        if mmd_weight < 0:
            raise ValueError("mmd_weight must be non-negative.")
        if mmd_bandwidths is None:
            mmd_bandwidths = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        if not mmd_bandwidths:
            raise ValueError("mmd_bandwidths must not be empty.")
        if any(bandwidth <= 0 for bandwidth in mmd_bandwidths):
            raise ValueError("mmd_bandwidths must contain only positive values.")
        kwargs.setdefault("kl_weight", 0.1)
        kwargs.setdefault("free_bits", 0.02)
        kwargs.setdefault("kl_warmup_epochs", 20)
        kwargs.setdefault("kl_start_weight", 0.0)
        kwargs.setdefault("use_mean_in_eval", True)
        self.mmd_weight = mmd_weight
        self.mmd_bandwidths = list(mmd_bandwidths)
        super().__init__(**kwargs)
