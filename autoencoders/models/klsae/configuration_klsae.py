"""Configuration for KL sparse autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class KLSparseAutoencoderConfig(AutoencoderConfig):
    """Configuration for a KL-sparse autoencoder."""

    model_type = "kl_sparse_autoencoder"

    def __init__(
        self,
        sparsity_weight: float = 1e-3,
        target_activation: float = 0.05,
        **kwargs,
    ) -> None:
        if sparsity_weight < 0:
            raise ValueError("sparsity_weight must be non-negative.")
        if not 0 < target_activation < 1:
            raise ValueError("target_activation must be between 0 and 1.")
        self.sparsity_weight = sparsity_weight
        self.target_activation = target_activation
        super().__init__(**kwargs)
