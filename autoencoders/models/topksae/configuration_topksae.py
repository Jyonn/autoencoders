"""Configuration for top-k sparse autoencoders."""

from __future__ import annotations

from ..ae.configuration_ae import AutoencoderConfig


class TopKSparseAutoencoderConfig(AutoencoderConfig):
    """Configuration for a top-k sparse autoencoder."""

    model_type = "topk_sparse_autoencoder"

    def __init__(
        self,
        topk: int = 4,
        **kwargs,
    ) -> None:
        latent_dim = kwargs.get("latent_dim")
        if latent_dim is None:
            raise TypeError("TopKSparseAutoencoderConfig requires `latent_dim`.")
        if topk <= 0:
            raise ValueError("topk must be a positive integer.")
        if topk > latent_dim:
            raise ValueError("topk must be less than or equal to latent_dim.")
        self.topk = topk
        super().__init__(**kwargs)
