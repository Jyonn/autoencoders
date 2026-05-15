"""Configuration for Beta-TCVAE models."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class BetaTCVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a Beta-TCVAE."""

    model_type = "beta_tc_variational_autoencoder"

    def __init__(
        self,
        mutual_information_weight: float = 1.0,
        total_correlation_weight: float = 6.0,
        dimension_wise_kl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if mutual_information_weight < 0:
            raise ValueError("mutual_information_weight must be non-negative.")
        if total_correlation_weight < 0:
            raise ValueError("total_correlation_weight must be non-negative.")
        if dimension_wise_kl_weight < 0:
            raise ValueError("dimension_wise_kl_weight must be non-negative.")
        kwargs.setdefault("kl_weight", 1.0)
        kwargs.setdefault("free_bits", 0.02)
        kwargs.setdefault("kl_warmup_epochs", 20)
        kwargs.setdefault("kl_start_weight", 0.0)
        kwargs.setdefault("use_mean_in_eval", True)
        self.mutual_information_weight = mutual_information_weight
        self.total_correlation_weight = total_correlation_weight
        self.dimension_wise_kl_weight = dimension_wise_kl_weight
        super().__init__(**kwargs)
