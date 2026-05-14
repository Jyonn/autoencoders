"""PyTorch implementation of a vector-quantized autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base.modeling_vq import BaseVectorQuantizedAutoencoderModel
from .configuration_vqvae import VectorQuantizedAutoencoderConfig


class VectorQuantizedAutoencoderModel(BaseVectorQuantizedAutoencoderModel):
    """A simple MLP VQ-VAE for vector-like feature inputs."""

    config_class = VectorQuantizedAutoencoderConfig
    min_input_rank = 3

    def iter_codebook_index_sets(self, codebook_indices: torch.Tensor) -> list[torch.Tensor]:
        if codebook_indices.ndim == 2:
            return [codebook_indices.reshape(-1)]
        return super().iter_codebook_index_sets(codebook_indices)

    def __init__(
        self,
        config: VectorQuantizedAutoencoderConfig,
        encoder=None,
        decoder=None,
        encoder_config=None,
        decoder_config=None,
    ) -> None:
        super().__init__(
            config,
            encoder=encoder,
            decoder=decoder,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
        self.codebook = nn.Embedding(self.config.codebook_size, self.config.latent_dim)
        self.codebook.weight.requires_grad_(not self.config.use_ema_codebook)
        self.register_buffer("ema_cluster_size", torch.zeros(self.config.codebook_size))
        self.register_buffer("ema_weight_sum", torch.zeros(self.config.codebook_size, self.config.latent_dim))
        self._reset_codebook()

    def _reset_codebook(self) -> None:
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / self.config.codebook_size,
            1.0 / self.config.codebook_size,
        )
        self.ema_cluster_size.fill_(1.0)
        self.ema_weight_sum.copy_(self.codebook.weight.detach())

    def _update_ema_codebook(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        flat_encoded = encoded.reshape(-1, encoded.shape[-1])
        flat_indices = codebook_indices.reshape(-1)
        one_hot_assignments = F.one_hot(flat_indices, num_classes=self.config.codebook_size).to(encoded.dtype)
        cluster_size_update = one_hot_assignments.sum(dim=0)
        embedding_sum_update = one_hot_assignments.transpose(0, 1) @ flat_encoded

        self.ema_cluster_size.mul_(self.config.ema_decay).add_(cluster_size_update, alpha=1 - self.config.ema_decay)
        self.ema_weight_sum.mul_(self.config.ema_decay).add_(embedding_sum_update, alpha=1 - self.config.ema_decay)

        n = self.ema_cluster_size.sum()
        normalized_cluster_size = (
            (self.ema_cluster_size + self.config.ema_epsilon)
            / (n + self.config.codebook_size * self.config.ema_epsilon)
            * n
        )
        normalized_embeddings = self.ema_weight_sum / normalized_cluster_size.unsqueeze(-1)
        self.codebook.weight.data.copy_(normalized_embeddings)

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        dead_code_mask = dead_code_mask.to(device=self.codebook.weight.device, dtype=torch.bool)
        dead_count = int(dead_code_mask.sum().item())
        if dead_count == 0:
            return 0

        if reference_latents is not None and reference_latents.numel() > 0:
            reference_latents = reference_latents.detach().to(self.codebook.weight.device).reshape(-1, self.config.latent_dim)
            sample_indices = torch.randint(0, reference_latents.shape[0], (dead_count,), device=self.codebook.weight.device)
            replacement_embeddings = reference_latents[sample_indices]
        else:
            replacement_embeddings = torch.empty(
                dead_count,
                self.config.latent_dim,
                device=self.codebook.weight.device,
            )
            replacement_embeddings.uniform_(
                -1.0 / self.config.codebook_size,
                1.0 / self.config.codebook_size,
            )

        self.codebook.weight.data[dead_code_mask] = replacement_embeddings
        if self.config.use_ema_codebook:
            self.ema_weight_sum.data[dead_code_mask] = replacement_embeddings
            self.ema_cluster_size.data[dead_code_mask] = 1.0
        return dead_count

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = (
            encoded.pow(2).sum(dim=-1, keepdim=True)
            - 2 * encoded @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1)
        )
        codebook_indices = distances.argmin(dim=-1)
        quantized_latents = self.codebook(codebook_indices)
        return quantized_latents, codebook_indices

    def on_quantizer_training_step(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        self._update_ema_codebook(encoded, codebook_indices)

    def get_quantized_export_extras(self) -> dict[str, object]:
        extras = super().get_quantized_export_extras()
        extras["codebook"] = self.codebook.weight.detach().clone()
        return extras
