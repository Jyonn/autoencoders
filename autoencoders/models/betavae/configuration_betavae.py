"""Configuration for beta variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class BetaVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a beta-VAE."""

    model_type = "beta_variational_autoencoder"

    def __init__(
        self,
        beta: float = 4.0,
        free_bits: float = 0.02,
        use_mean_in_eval: bool = True,
        **kwargs,
    ) -> None:
        kl_weight = kwargs.pop("kl_weight", beta)

        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative.")
        kwargs["kl_weight"] = kl_weight
        kwargs.setdefault("free_bits", free_bits)
        kwargs.setdefault("use_mean_in_eval", use_mean_in_eval)
        self.beta = kl_weight
        super().__init__(**kwargs)
