"""PyTorch implementation of a product-quantized autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base.modeling_vq import BaseVectorQuantizedAutoencoderModel
from .configuration_pqvae import ProductQuantizedAutoencoderConfig


class ProductQuantizedAutoencoderModel(BaseVectorQuantizedAutoencoderModel):
    """An MLP autoencoder with product quantization in latent space."""

    config_class = ProductQuantizedAutoencoderConfig
    config: ProductQuantizedAutoencoderConfig

    def __init__(
        self,
        config: ProductQuantizedAutoencoderConfig,
        **kwargs: object,
    ) -> None:
        super().__init__(config, **kwargs)
        self.subspace_dim = self.config.latent_dim // self.config.num_codebooks
        self.codebooks = nn.ModuleList(
            [nn.Embedding(self.config.codebook_size, self.subspace_dim) for _ in range(self.config.num_codebooks)]
        )
        for codebook in self.codebooks:
            codebook.weight.requires_grad_(not self.config.use_ema_codebook)

        self.register_buffer("ema_cluster_size", torch.zeros(self.config.num_codebooks, self.config.codebook_size))
        self.register_buffer(
            "ema_weight_sum",
            torch.zeros(self.config.num_codebooks, self.config.codebook_size, self.subspace_dim),
        )
        self._reset_codebooks()

    def _reset_codebooks(self) -> None:
        for codebook in self.codebooks:
            nn.init.uniform_(
                codebook.weight,
                -1.0 / self.config.codebook_size,
                1.0 / self.config.codebook_size,
            )
        self.ema_cluster_size.fill_(1.0)
        self.ema_weight_sum.copy_(torch.stack([codebook.weight.detach() for codebook in self.codebooks], dim=0))

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_groups = encoded.view(encoded.shape[0], self.config.num_codebooks, self.subspace_dim)
        quantized_groups: list[torch.Tensor] = []
        index_groups: list[torch.Tensor] = []

        for codebook_index, codebook in enumerate(self.codebooks):
            encoded_group = encoded_groups[:, codebook_index, :]
            distances = (
                encoded_group.pow(2).sum(dim=-1, keepdim=True)
                - 2 * encoded_group @ codebook.weight.t()
                + codebook.weight.pow(2).sum(dim=-1)
            )
            indices = distances.argmin(dim=-1)
            quantized_groups.append(codebook(indices))
            index_groups.append(indices)

        quantized = torch.stack(quantized_groups, dim=1)
        indices = torch.stack(index_groups, dim=1)
        return quantized.view(encoded.shape[0], self.config.latent_dim), indices

    def _update_ema_codebooks(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        encoded_groups = encoded.view(encoded.shape[0], self.config.num_codebooks, self.subspace_dim)
        for codebook_index, codebook in enumerate(self.codebooks):
            encoded_group = encoded_groups[:, codebook_index, :]
            indices = codebook_indices[:, codebook_index]
            one_hot = F.one_hot(indices, num_classes=self.config.codebook_size).to(encoded_group.dtype)
            cluster_size_update = one_hot.sum(dim=0)
            embedding_sum_update = one_hot.transpose(0, 1) @ encoded_group

            self.ema_cluster_size[codebook_index].mul_(self.config.ema_decay).add_(
                cluster_size_update,
                alpha=1 - self.config.ema_decay,
            )
            self.ema_weight_sum[codebook_index].mul_(self.config.ema_decay).add_(
                embedding_sum_update,
                alpha=1 - self.config.ema_decay,
            )

            n = self.ema_cluster_size[codebook_index].sum()
            normalized_cluster_size = (
                (self.ema_cluster_size[codebook_index] + self.config.ema_epsilon)
                / (n + self.config.codebook_size * self.config.ema_epsilon)
                * n
            )
            normalized_embeddings = self.ema_weight_sum[codebook_index] / normalized_cluster_size.unsqueeze(-1)
            codebook.weight.data.copy_(normalized_embeddings)

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        dead_code_mask = dead_code_mask.to(device=self.codebooks[0].weight.device, dtype=torch.bool)
        dead_count = int(dead_code_mask.sum().item())
        if dead_count == 0:
            return 0

        reference_groups = None
        if reference_latents is not None and reference_latents.numel() > 0:
            reference_groups = reference_latents.detach().to(self.codebooks[0].weight.device)
            reference_groups = reference_groups.view(reference_groups.shape[0], self.config.num_codebooks, self.subspace_dim)

        for codebook_index, codebook in enumerate(self.codebooks):
            mask = dead_code_mask[codebook_index]
            count = int(mask.sum().item())
            if count == 0:
                continue
            if reference_groups is not None:
                sample_indices = torch.randint(
                    0,
                    reference_groups.shape[0],
                    (count,),
                    device=codebook.weight.device,
                )
                replacements = reference_groups[sample_indices, codebook_index, :]
            else:
                replacements = torch.empty(count, self.subspace_dim, device=codebook.weight.device)
                replacements.uniform_(-1.0 / self.config.codebook_size, 1.0 / self.config.codebook_size)

            codebook.weight.data[mask] = replacements
            if self.config.use_ema_codebook:
                self.ema_weight_sum.data[codebook_index, mask] = replacements
                self.ema_cluster_size.data[codebook_index, mask] = 1.0

        return dead_count

    def on_quantizer_training_step(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        self._update_ema_codebooks(encoded, codebook_indices)

    def get_quantized_export_extras(self) -> dict[str, object]:
        extras = super().get_quantized_export_extras()
        extras["num_codebooks"] = self.config.num_codebooks
        extras["codebooks"] = torch.stack([codebook.weight.detach().clone() for codebook in self.codebooks], dim=0)
        return extras
