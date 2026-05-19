"""PyTorch implementation of a residual-quantized autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...function import kmeans_cluster_centers
from ..base.modeling_vq import BaseVectorQuantizedAutoencoderModel
from .configuration_rqvae import ResidualQuantizedAutoencoderConfig


class ResidualQuantizedAutoencoderModel(BaseVectorQuantizedAutoencoderModel):
    """An MLP autoencoder with residual vector quantization."""

    config_class = ResidualQuantizedAutoencoderConfig
    config: ResidualQuantizedAutoencoderConfig

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._reference_residual_batches: list[torch.Tensor] = []
        self._latest_commitment_loss: torch.Tensor | None = None
        self._latest_codebook_loss: torch.Tensor | None = None
        self.codebooks = nn.ModuleList(
            [nn.Embedding(self.config.codebook_size, self.config.latent_dim) for _ in range(self.config.num_quantizers)]
        )
        for codebook in self.codebooks:
            codebook.weight.requires_grad_(not self.config.use_ema_codebook)

        self.register_buffer("ema_cluster_size", torch.zeros(self.config.num_quantizers, self.config.codebook_size))
        self.register_buffer(
            "ema_weight_sum",
            torch.zeros(self.config.num_quantizers, self.config.codebook_size, self.config.latent_dim),
        )
        self._reset_codebooks()

    def _reset_codebooks(self) -> None:
        if self.config.kmeans_init:
            for codebook in self.codebooks:
                codebook.weight.data.zero_()
            self.ema_cluster_size.zero_()
            self.ema_weight_sum.zero_()
            return
        for codebook in self.codebooks:
            nn.init.uniform_(
                codebook.weight,
                -1.0 / self.config.codebook_size,
                1.0 / self.config.codebook_size,
            )
        self.ema_cluster_size.fill_(1.0)
        self.ema_weight_sum.copy_(torch.stack([codebook.weight.detach() for codebook in self.codebooks], dim=0))

    def initialize_codebooks(self, encoded: torch.Tensor) -> None:
        residual = encoded.reshape(-1, self.config.latent_dim)
        initialized_codebooks: list[torch.Tensor] = []

        for codebook_index, codebook in enumerate(self.codebooks):
            centers = kmeans_cluster_centers(
                residual,
                self.config.codebook_size,
                self.config.kmeans_iters,
            )
            codebook.weight.data.copy_(centers)
            initialized_codebooks.append(centers)

            distances = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2 * residual @ centers.t()
                + centers.pow(2).sum(dim=-1)
            )
            indices = self.assign_codebook_indices_for_slot(distances, slot=codebook_index)
            residual = residual - centers[indices]

        self.ema_cluster_size.fill_(1.0)
        self.ema_weight_sum.copy_(torch.stack(initialized_codebooks, dim=0))

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = encoded
        quantized_sum = torch.zeros_like(encoded)
        index_steps: list[torch.Tensor] = []
        commitment_losses: list[torch.Tensor] = []
        codebook_losses: list[torch.Tensor] = []

        for codebook_index, codebook in enumerate(self.codebooks):
            distances = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2 * residual @ codebook.weight.t()
                + codebook.weight.pow(2).sum(dim=-1)
            )
            indices = self.assign_codebook_indices_for_slot(distances, slot=codebook_index)
            quantized = codebook(indices)
            commitment_losses.append(F.mse_loss(residual, quantized.detach()))
            codebook_losses.append(F.mse_loss(quantized, residual.detach()))
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            index_steps.append(indices)

        self._latest_commitment_loss = torch.stack(commitment_losses).mean()
        self._latest_codebook_loss = torch.stack(codebook_losses).mean()
        return quantized_sum, torch.stack(index_steps, dim=1)

    def compute_commitment_loss(self, encoded: torch.Tensor, quantized_latents: torch.Tensor) -> torch.Tensor:
        if self._latest_commitment_loss is None:
            return super().compute_commitment_loss(encoded, quantized_latents)
        return self._latest_commitment_loss

    def compute_codebook_loss(self, encoded: torch.Tensor, quantized_latents: torch.Tensor) -> torch.Tensor:
        if self._latest_codebook_loss is None:
            return super().compute_codebook_loss(encoded, quantized_latents)
        return self._latest_codebook_loss

    def _update_ema_codebooks(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        residual = encoded
        for codebook_index, codebook in enumerate(self.codebooks):
            indices = codebook_indices[:, codebook_index]
            quantized = codebook(indices).detach()
            one_hot = F.one_hot(indices, num_classes=self.config.codebook_size).to(encoded.dtype)
            cluster_size_update = one_hot.sum(dim=0)
            embedding_sum_update = one_hot.transpose(0, 1) @ residual

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
            # Keep the residual chain aligned with the forward pass: later
            # residual quantizers should see the pre-update quantized vectors
            # that produced the recorded indices, not freshly updated codes.
            residual = residual - quantized

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        dead_code_mask = dead_code_mask.to(device=self.codebooks[0].weight.device, dtype=torch.bool)
        dead_count = int(dead_code_mask.sum().item())
        if dead_count == 0:
            return 0

        reference_latents = None if reference_latents is None else reference_latents.detach().to(self.codebooks[0].weight.device)
        for codebook_index, codebook in enumerate(self.codebooks):
            mask = dead_code_mask[codebook_index]
            count = int(mask.sum().item())
            if count == 0:
                continue
            if reference_latents is not None and reference_latents.numel() > 0:
                if reference_latents.ndim == 3:
                    candidates = reference_latents[:, codebook_index, :]
                else:
                    candidates = reference_latents
                sample_indices = torch.randint(0, candidates.shape[0], (count,), device=codebook.weight.device)
                replacements = candidates[sample_indices]
            else:
                replacements = torch.empty(count, self.config.latent_dim, device=codebook.weight.device)
                replacements.uniform_(-1.0 / self.config.codebook_size, 1.0 / self.config.codebook_size)

            codebook.weight.data[mask] = replacements
            if self.config.use_ema_codebook:
                self.ema_weight_sum.data[codebook_index, mask] = replacements
                self.ema_cluster_size.data[codebook_index, mask] = 1.0

        return dead_count

    def on_quantizer_training_step(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        self._update_ema_codebooks(encoded, codebook_indices)

    def _compute_reference_residuals(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> torch.Tensor:
        residual = encoded
        residual_steps: list[torch.Tensor] = []
        for codebook_index, codebook in enumerate(self.codebooks):
            residual_steps.append(residual.detach())
            indices = codebook_indices[:, codebook_index]
            quantized = codebook(indices).detach()
            residual = residual - quantized
        return torch.stack(residual_steps, dim=1)

    def _maybe_reset_dead_codes(
        self,
        *,
        encoded: torch.Tensor,
        codebook_indices: torch.Tensor,
        is_last_train_step: bool | None,
    ) -> None:
        self._last_dead_code_reset_count = 0
        if not self.training or not self.config.dead_code_reset:
            return
        if is_last_train_step is None:
            raise ValueError("is_last_train_step must be provided when dead_code_reset is enabled during training.")

        self._accumulate_code_usage(codebook_indices.detach())
        self._reference_residual_batches.append(self._compute_reference_residuals(encoded.detach(), codebook_indices.detach()))
        if not is_last_train_step:
            return

        dead_code_mask = self._code_usage_counts <= self.config.dead_code_threshold
        reference_latents = (
            torch.cat(self._reference_residual_batches, dim=0)
            if self._reference_residual_batches
            else None
        )
        self._last_dead_code_reset_count = int(self.reset_dead_codes(dead_code_mask, reference_latents))
        self._code_usage_counts.zero_()
        self._reference_residual_batches.clear()

    def get_quantized_export_extras(self) -> dict[str, object]:
        extras = super().get_quantized_export_extras()
        extras["num_quantizers"] = self.config.num_quantizers
        extras["codebooks"] = torch.stack([codebook.weight.detach().clone() for codebook in self.codebooks], dim=0)
        return extras
