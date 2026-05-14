"""Configuration for hierarchical variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class HierarchicalVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a hierarchical variational autoencoder."""

    model_type = "hierarchical_variational_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        kl_weight: float = 1.0,
        free_bits: float = 0.02,
        use_mean_in_eval: bool = True,
        top_latent_dim: int | None = None,
        **kwargs,
    ) -> None:
        top_latent_dim = latent_dim if top_latent_dim is None else top_latent_dim
        if top_latent_dim <= 0:
            raise ValueError("top_latent_dim must be a positive integer.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            free_bits=free_bits,
            use_mean_in_eval=use_mean_in_eval,
            top_latent_dim=top_latent_dim,
            **kwargs,
        )
