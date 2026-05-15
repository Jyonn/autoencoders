"""Configuration for adversarial autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class AdversarialAutoencoderConfig(AutoencoderConfig):
    """Configuration for an adversarial autoencoder."""

    model_type = "adversarial_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        adversarial_weight: float = 1.0,
        discriminator_hidden_dims: list[int] | None = None,
        **kwargs,
    ) -> None:
        if adversarial_weight < 0:
            raise ValueError("adversarial_weight must be non-negative.")
        if discriminator_hidden_dims is None:
            discriminator_hidden_dims = [128, 64]
        if any(dim <= 0 for dim in discriminator_hidden_dims):
            raise ValueError("discriminator_hidden_dims must contain positive integers.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            adversarial_weight=adversarial_weight,
            discriminator_hidden_dims=list(discriminator_hidden_dims),
            **kwargs,
        )
