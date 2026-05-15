"""Configuration for FactorVAE models."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class FactorVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a FactorVAE."""

    model_type = "factor_variational_autoencoder"

    def __init__(
        self,
        tc_weight: float = 10.0,
        discriminator_hidden_dims: list[int] | None = None,
        **kwargs,
    ) -> None:
        if tc_weight < 0:
            raise ValueError("tc_weight must be non-negative.")
        if discriminator_hidden_dims is None:
            discriminator_hidden_dims = [128, 64]
        if any(dim <= 0 for dim in discriminator_hidden_dims):
            raise ValueError("discriminator_hidden_dims must contain positive integers.")
        kwargs.setdefault("kl_weight", 0.1)
        kwargs.setdefault("free_bits", 0.02)
        kwargs.setdefault("kl_warmup_epochs", 20)
        kwargs.setdefault("kl_start_weight", 0.0)
        kwargs.setdefault("use_mean_in_eval", True)
        self.tc_weight = tc_weight
        self.discriminator_hidden_dims = list(discriminator_hidden_dims)
        super().__init__(**kwargs)
