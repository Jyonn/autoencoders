"""Configuration for DIP-VAE models."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class DIPVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a DIP-VAE."""

    model_type = "dip_variational_autoencoder"

    def __init__(
        self,
        dip_weight: float = 10.0,
        dip_offdiag_weight: float = 1.0,
        dip_diag_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if dip_weight < 0:
            raise ValueError("dip_weight must be non-negative.")
        if dip_offdiag_weight < 0:
            raise ValueError("dip_offdiag_weight must be non-negative.")
        if dip_diag_weight < 0:
            raise ValueError("dip_diag_weight must be non-negative.")
        kwargs.setdefault("kl_weight", 0.1)
        kwargs.setdefault("free_bits", 0.02)
        kwargs.setdefault("kl_warmup_epochs", 20)
        kwargs.setdefault("kl_start_weight", 0.0)
        kwargs.setdefault("use_mean_in_eval", True)
        self.dip_weight = dip_weight
        self.dip_offdiag_weight = dip_offdiag_weight
        self.dip_diag_weight = dip_diag_weight
        super().__init__(**kwargs)
