"""Configuration for residual finite-scalar quantized autoencoders."""

from __future__ import annotations

from ..fsq.configuration_fsq import FiniteScalarQuantizedAutoencoderConfig


class ResidualFiniteScalarQuantizedAutoencoderConfig(FiniteScalarQuantizedAutoencoderConfig):
    """Configuration for a residual FSQ autoencoder."""

    model_type = "residual_finite_scalar_quantized_autoencoder"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
        reconstruction_loss: str = "mse",
        num_levels: int = 8,
        commitment_weight: float = 0.25,
        quantization_bound: float = 1.0,
        num_quantizers: int = 2,
        **kwargs,
    ) -> None:
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive.")
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation,
            use_bias=use_bias,
            reconstruction_loss=reconstruction_loss,
            num_levels=num_levels,
            commitment_weight=commitment_weight,
            quantization_bound=quantization_bound,
            num_quantizers=num_quantizers,
            **kwargs,
        )
