"""Configuration for denoising variational autoencoders."""

from __future__ import annotations

from ..vae.configuration_vae import VariationalAutoencoderConfig


class DenoisingVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """Configuration for a denoising variational autoencoder."""

    model_type = "denoising_variational_autoencoder"

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
        use_mean_in_eval: bool = True,
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
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            use_mean_in_eval=use_mean_in_eval,
            noise_type=noise_type,
            noise_std=noise_std,
            masking_ratio=masking_ratio,
            apply_noise_in_eval=apply_noise_in_eval,
            **kwargs,
        )
