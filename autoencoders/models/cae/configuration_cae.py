"""Configuration for contractive autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class ContractiveAutoencoderConfig(AutoencoderConfig):
    """Configuration for a contractive autoencoder."""

    model_type = "contractive_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        contractive_weight: float = 1e-2,
        **kwargs,
    ) -> None:
        if contractive_weight < 0:
            raise ValueError("contractive_weight must be non-negative.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            contractive_weight=contractive_weight,
            **kwargs,
        )
