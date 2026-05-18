"""Configuration for hierarchical vector-quantized autoencoders."""

from __future__ import annotations

from ..base.configuration_vq import BaseVectorQuantizedAutoencoderConfig


class HierarchicalVectorQuantizedAutoencoderConfig(BaseVectorQuantizedAutoencoderConfig):
    """Configuration for a two-level VQ-VAE-2 style autoencoder."""

    model_type = "hierarchical_vector_quantized_autoencoder"

    def __init__(
        self,
        top_latent_dim: int | None = None,
        **kwargs,
    ) -> None:
        latent_dim = kwargs.get("latent_dim")
        if latent_dim is None:
            raise TypeError("HierarchicalVectorQuantizedAutoencoderConfig requires `latent_dim`.")
        if top_latent_dim is None:
            top_latent_dim = latent_dim
        if top_latent_dim <= 0:
            raise ValueError("top_latent_dim must be positive.")
        self.top_latent_dim = top_latent_dim
        super().__init__(**kwargs)
        self.validate_sinkhorn_slot_count(2, "HierarchicalVectorQuantizedAutoencoderConfig")
