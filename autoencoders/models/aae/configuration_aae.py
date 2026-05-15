"""Configuration for adversarial autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class AdversarialAutoencoderConfig(AutoencoderConfig):
    """Configuration for an adversarial autoencoder."""

    model_type = "adversarial_autoencoder"

    def __init__(
        self,
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
        self.adversarial_weight = adversarial_weight
        self.discriminator_hidden_dims = list(discriminator_hidden_dims)
        super().__init__(**kwargs)
