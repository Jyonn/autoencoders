"""Configuration for denoising variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class DenoisingVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a denoising variational autoencoder."""

    model_type = "denoising_variational_autoencoder"

    def __init__(
        self,
        noise_type: str = "gaussian",
        noise_std: float = 0.1,
        masking_ratio: float = 0.3,
        apply_noise_in_eval: bool = False,
        **kwargs,
    ) -> None:
        if noise_type not in {"gaussian", "masking"}:
            raise ValueError("noise_type must be one of: 'gaussian', 'masking'.")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative.")
        if not 0 <= masking_ratio <= 1:
            raise ValueError("masking_ratio must be between 0 and 1.")
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.masking_ratio = masking_ratio
        self.apply_noise_in_eval = apply_noise_in_eval
        super().__init__(**kwargs)
