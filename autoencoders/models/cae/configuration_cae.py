"""Configuration for contractive autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class ContractiveAutoencoderConfig(AutoencoderConfig):
    """Configuration for a contractive autoencoder."""

    model_type = "contractive_autoencoder"

    def __init__(
        self,
        contractive_weight: float = 1e-2,
        **kwargs,
    ) -> None:
        if contractive_weight < 0:
            raise ValueError("contractive_weight must be non-negative.")
        self.contractive_weight = contractive_weight
        super().__init__(**kwargs)
