"""Configuration for denoising autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class DenoisingAutoencoderConfig(AutoencoderConfig):
    """Configuration for a denoising autoencoder."""

    model_type = "denoising_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reconstruction_loss: str = "mse",
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

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            noise_type=noise_type,
            noise_std=noise_std,
            masking_ratio=masking_ratio,
            apply_noise_in_eval=apply_noise_in_eval,
            **kwargs,
        )
