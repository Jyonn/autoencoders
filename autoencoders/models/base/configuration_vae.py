"""Base configuration shared by variational autoencoder families."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class BaseVariationalAutoencoderConfig(AutoencoderConfig):
    """Base config for VAE-style autoencoders."""

    model_type = "base_variational_autoencoder"

    def __init__(
        self,
        kl_weight: float = 1.0,
        free_bits: float = 0.02,
        kl_warmup_epochs: int = 0,
        kl_start_weight: float = 0.0,
        use_mean_in_eval: bool = True,
        **kwargs,
    ) -> None:
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative.")
        if free_bits < 0:
            raise ValueError("free_bits must be non-negative.")
        if kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be greater than or equal to 0.")
        if kl_start_weight < 0:
            raise ValueError("kl_start_weight must be non-negative.")
        self.kl_weight = kl_weight
        self.free_bits = free_bits
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_start_weight = kl_start_weight
        self.use_mean_in_eval = use_mean_in_eval
        super().__init__(**kwargs)
