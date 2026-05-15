"""Configuration for variational autoencoders with a learned VampPrior."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class VampPriorVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a VampPrior variational autoencoder."""

    model_type = "vamp_prior_variational_autoencoder"

    def __init__(
        self,
        num_pseudo_inputs: int = 128,
        pseudo_input_std: float = 0.01,
        **kwargs,
    ) -> None:
        if num_pseudo_inputs <= 0:
            raise ValueError("num_pseudo_inputs must be positive.")
        if pseudo_input_std < 0:
            raise ValueError("pseudo_input_std must be non-negative.")
        kwargs.setdefault("kl_weight", 0.1)
        kwargs.setdefault("free_bits", 0.02)
        kwargs.setdefault("kl_warmup_epochs", 20)
        kwargs.setdefault("kl_start_weight", 0.0)
        kwargs.setdefault("use_mean_in_eval", True)
        self.num_pseudo_inputs = num_pseudo_inputs
        self.pseudo_input_std = pseudo_input_std
        super().__init__(**kwargs)
