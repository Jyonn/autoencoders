"""Configuration for variational autoencoders with a learned VampPrior."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class VampPriorVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a VampPrior variational autoencoder."""

    model_type = "vamp_prior_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        kl_weight: float = 0.1,
        free_bits: float = 0.02,
        kl_warmup_epochs: int = 20,
        kl_start_weight: float = 0.0,
        use_mean_in_eval: bool = True,
        num_pseudo_inputs: int = 128,
        pseudo_input_std: float = 0.01,
        **kwargs,
    ) -> None:
        if num_pseudo_inputs <= 0:
            raise ValueError("num_pseudo_inputs must be positive.")
        if pseudo_input_std < 0:
            raise ValueError("pseudo_input_std must be non-negative.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_start_weight=kl_start_weight,
            use_mean_in_eval=use_mean_in_eval,
            num_pseudo_inputs=num_pseudo_inputs,
            pseudo_input_std=pseudo_input_std,
            **kwargs,
        )
