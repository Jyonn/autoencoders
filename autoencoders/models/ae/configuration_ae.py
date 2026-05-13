"""Configuration for the basic deterministic autoencoder."""

from __future__ import annotations

from ..base.configuration_base import BaseAutoencoderConfig


class AutoencoderConfig(BaseAutoencoderConfig):
    """Configuration for a feed-forward autoencoder."""

    model_type = "autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        **kwargs,
    ) -> None:
        hidden_dims = [256, 128] if hidden_dims is None else list(hidden_dims)
        decoder_hidden_dims = None if decoder_hidden_dims is None else list(decoder_hidden_dims)

        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if decoder_hidden_dims is not None and any(dim <= 0 for dim in decoder_hidden_dims):
            raise ValueError("decoder_hidden_dims must contain positive integers.")
        if activation not in {"relu", "gelu", "silu", "tanh"}:
            raise ValueError("activation must be one of: 'relu', 'gelu', 'silu', 'tanh'.")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss=reconstruction_loss,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )

