"""Configuration for sparse autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class SparseAutoencoderConfig(AutoencoderConfig):
    """Configuration for a sparse autoencoder."""

    model_type = "sparse_autoencoder"

    def __init__(
        self,
        sparsity_weight: float = 1e-3,
        **kwargs,
    ) -> None:
        if sparsity_weight < 0:
            raise ValueError("sparsity_weight must be non-negative.")
        self.sparsity_weight = sparsity_weight
        super().__init__(**kwargs)
