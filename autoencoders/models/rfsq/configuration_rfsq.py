"""Configuration for residual finite-scalar quantized autoencoders."""

from __future__ import annotations

from ..fsq.configuration_fsq import FiniteScalarQuantizedAutoencoderConfig


class ResidualFiniteScalarQuantizedAutoencoderConfig(FiniteScalarQuantizedAutoencoderConfig):
    """Configuration for a residual FSQ autoencoder."""

    model_type = "residual_finite_scalar_quantized_autoencoder"

    def __init__(
        self,
        num_quantizers: int = 2,
        **kwargs,
    ) -> None:
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive.")
        self.num_quantizers = num_quantizers
        super().__init__(**kwargs)
