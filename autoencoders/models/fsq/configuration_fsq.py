"""Configuration for finite scalar quantized autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class FiniteScalarQuantizedAutoencoderConfig(AutoencoderConfig):
    """Configuration for a finite scalar quantized autoencoder."""

    model_type = "finite_scalar_quantized_autoencoder"

    def __init__(
        self,
        num_levels: int = 8,
        commitment_weight: float = 0.25,
        quantization_bound: float = 1.0,
        **kwargs,
    ) -> None:
        if num_levels <= 1:
            raise ValueError("num_levels must be greater than 1.")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative.")
        if quantization_bound <= 0:
            raise ValueError("quantization_bound must be positive.")
        kwargs.pop("codebook_size", None)
        kwargs["codebook_size"] = num_levels
        self.num_levels = num_levels
        self.commitment_weight = commitment_weight
        self.quantization_bound = quantization_bound
        super().__init__(**kwargs)
