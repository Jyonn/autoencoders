"""Configuration for product-quantized autoencoders."""

from __future__ import annotations

from ..vqvae.configuration_vqvae import VectorQuantizedAutoencoderConfig


class ProductQuantizedAutoencoderConfig(VectorQuantizedAutoencoderConfig):
    """Configuration for a product-quantized autoencoder."""

    model_type = "product_quantized_autoencoder"

    def __init__(
        self,
        num_codebooks: int = 2,
        **kwargs,
    ) -> None:
        latent_dim = kwargs.get("latent_dim")
        if latent_dim is None:
            raise TypeError("ProductQuantizedAutoencoderConfig requires `latent_dim`.")
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be a positive integer.")
        if latent_dim % num_codebooks != 0:
            raise ValueError("latent_dim must be divisible by num_codebooks.")
        self.num_codebooks = num_codebooks
        super().__init__(**kwargs)
        self.validate_sinkhorn_slot_count(self.num_codebooks, "ProductQuantizedAutoencoderConfig")
