"""PyTorch implementation of a VQ-VAE-2 style hierarchical autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import HierarchicalQuantizedAutoencoderOutput
from ..base.modeling_vq import BaseVectorQuantizedAutoencoderModel
from .configuration_vqvae2 import HierarchicalVectorQuantizedAutoencoderConfig


class HierarchicalVectorQuantizedAutoencoderModel(BaseVectorQuantizedAutoencoderModel):
    """A two-level vector quantized autoencoder for embedding inputs."""

    config_class = HierarchicalVectorQuantizedAutoencoderConfig
    min_input_rank = 3

    def __init__(self, config: HierarchicalVectorQuantizedAutoencoderConfig) -> None:
        super().__init__(config)
        self.top_encoder = nn.Linear(self.config.latent_dim, self.config.top_latent_dim, bias=self.config.use_bias)
        self.top_decoder = nn.Linear(self.config.top_latent_dim, self.config.latent_dim, bias=self.config.use_bias)
        self.decoder = self._build_hierarchical_decoder()
        self.top_codebook = nn.Embedding(self.config.codebook_size, self.config.top_latent_dim)
        self.bottom_codebook = nn.Embedding(self.config.codebook_size, self.config.latent_dim)
        self.top_codebook.weight.requires_grad_(not self.config.use_ema_codebook)
        self.bottom_codebook.weight.requires_grad_(not self.config.use_ema_codebook)
        self.register_buffer("ema_cluster_size", torch.zeros(2, self.config.codebook_size))
        self.register_buffer(
            "ema_weight_sum",
            torch.zeros(2, self.config.codebook_size, max(self.config.top_latent_dim, self.config.latent_dim)),
        )
        self._reset_codebooks()

    def _build_hierarchical_decoder(self) -> nn.Sequential:
        decoder_hidden_dims = self.config.decoder_hidden_dims
        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(self.config.hidden_dims))
        dims = [self.config.top_latent_dim + self.config.latent_dim, *decoder_hidden_dims, self.config.input_dim]
        return self._build_mlp(dims)

    def _build_mlp(self, dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        activation_factory = self._get_activation_factory()
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim, bias=self.config.use_bias))
            if index != len(dims) - 2:
                layers.append(activation_factory())
        return nn.Sequential(*layers)

    def _get_activation_factory(self):
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        return activations[self.config.activation]

    def _reset_codebooks(self) -> None:
        for codebook in (self.top_codebook, self.bottom_codebook):
            nn.init.uniform_(codebook.weight, -1.0 / self.config.codebook_size, 1.0 / self.config.codebook_size)
        self.ema_cluster_size.fill_(1.0)
        self.ema_weight_sum.zero_()
        self.ema_weight_sum[0, :, : self.config.top_latent_dim] = self.top_codebook.weight.detach()
        self.ema_weight_sum[1, :, : self.config.latent_dim] = self.bottom_codebook.weight.detach()

    def _quantize_with_codebook(self, encoded: torch.Tensor, codebook: nn.Embedding) -> tuple[torch.Tensor, torch.Tensor]:
        distances = (
            encoded.pow(2).sum(dim=-1, keepdim=True)
            - 2 * encoded @ codebook.weight.t()
            + codebook.weight.pow(2).sum(dim=-1)
        )
        indices = distances.argmin(dim=-1)
        quantized = codebook(indices)
        return quantized, indices

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top_encoded = self.top_encoder(encoded)
        top_quantized, top_indices = self._quantize_with_codebook(top_encoded, self.top_codebook)
        bottom_context = encoded + self.top_decoder(top_quantized)
        bottom_quantized, bottom_indices = self._quantize_with_codebook(bottom_context, self.bottom_codebook)
        hierarchical_latents = torch.cat([top_quantized, bottom_quantized], dim=-1)
        codebook_indices = torch.stack([top_indices, bottom_indices], dim=-1)
        return hierarchical_latents, codebook_indices

    def compute_codebook_loss(self, encoded: torch.Tensor, quantized_latents: torch.Tensor) -> torch.Tensor:
        del encoded, quantized_latents
        return self.top_codebook.weight.new_zeros(())

    def compute_commitment_loss_components(
        self,
        encoded: torch.Tensor,
        top_quantized: torch.Tensor,
        top_encoded: torch.Tensor,
        bottom_quantized: torch.Tensor,
        bottom_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_commitment_loss = F.mse_loss(top_encoded, top_quantized.detach())
        bottom_commitment_loss = F.mse_loss(bottom_context, bottom_quantized.detach())
        return top_commitment_loss, bottom_commitment_loss

    def on_quantizer_training_step(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        if not self.config.use_ema_codebook:
            return
        top_encoded = self.top_encoder(encoded)
        top_indices = codebook_indices[..., 0].reshape(-1)
        top_one_hot = F.one_hot(top_indices, num_classes=self.config.codebook_size).to(encoded.dtype)
        top_values = top_encoded.reshape(-1, self.config.top_latent_dim)
        self._update_ema_slot(0, top_one_hot, top_values, self.top_codebook, self.config.top_latent_dim)
        bottom_context = encoded + self.top_decoder(self.top_codebook(codebook_indices[..., 0]).detach())
        bottom_indices = codebook_indices[..., 1].reshape(-1)
        bottom_one_hot = F.one_hot(bottom_indices, num_classes=self.config.codebook_size).to(encoded.dtype)
        bottom_values = bottom_context.reshape(-1, self.config.latent_dim)
        self._update_ema_slot(1, bottom_one_hot, bottom_values, self.bottom_codebook, self.config.latent_dim)

    def _update_ema_slot(self, slot: int, one_hot: torch.Tensor, values: torch.Tensor, codebook: nn.Embedding, width: int) -> None:
        cluster_size_update = one_hot.sum(dim=0)
        embedding_sum_update = one_hot.transpose(0, 1) @ values
        self.ema_cluster_size[slot].mul_(self.config.ema_decay).add_(cluster_size_update, alpha=1 - self.config.ema_decay)
        self.ema_weight_sum[slot, :, :width].mul_(self.config.ema_decay).add_(
            embedding_sum_update,
            alpha=1 - self.config.ema_decay,
        )
        n = self.ema_cluster_size[slot].sum()
        normalized_cluster_size = (
            (self.ema_cluster_size[slot] + self.config.ema_epsilon)
            / (n + self.config.codebook_size * self.config.ema_epsilon)
            * n
        )
        codebook.weight.data.copy_(self.ema_weight_sum[slot, :, :width] / normalized_cluster_size.unsqueeze(-1))

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        dead_code_mask = dead_code_mask.to(device=self.top_codebook.weight.device, dtype=torch.bool)
        dead_count = int(dead_code_mask.sum().item())
        if dead_count == 0:
            return 0
        reference_latents = (
            None
            if reference_latents is None
            else reference_latents.detach().to(self.top_codebook.weight.device).reshape(-1, reference_latents.shape[-1])
        )
        codebooks = ((self.top_codebook, self.config.top_latent_dim, 0), (self.bottom_codebook, self.config.latent_dim, 1))
        for slot, (codebook, width, ema_slot) in enumerate(codebooks):
            mask = dead_code_mask[slot]
            count = int(mask.sum().item())
            if count == 0:
                continue
            if reference_latents is not None and reference_latents.numel() > 0:
                sample_indices = torch.randint(0, reference_latents.shape[0], (count,), device=codebook.weight.device)
                replacements = reference_latents[sample_indices, :width]
            else:
                replacements = torch.empty(count, width, device=codebook.weight.device)
                replacements.uniform_(-1.0 / self.config.codebook_size, 1.0 / self.config.codebook_size)
            codebook.weight.data[mask] = replacements
            if self.config.use_ema_codebook:
                self.ema_weight_sum.data[ema_slot, mask, :width] = replacements
                self.ema_cluster_size.data[ema_slot, mask] = 1.0
        return dead_count

    def get_quantized_export_extras(self) -> dict[str, object]:
        extras = super().get_quantized_export_extras()
        extras["top_latent_dim"] = self.config.top_latent_dim
        extras["top_codebook"] = self.top_codebook.weight.detach().clone()
        extras["bottom_codebook"] = self.bottom_codebook.weight.detach().clone()
        return extras

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        is_last_train_step: bool | None = None,
        **kwargs: object,
    ) -> HierarchicalQuantizedAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        self.validate_inputs(inputs)
        encoded = self.encode(inputs)
        top_encoded = self.top_encoder(encoded)
        top_quantized, top_indices = self._quantize_with_codebook(top_encoded, self.top_codebook)
        bottom_context = encoded + self.top_decoder(top_quantized)
        bottom_quantized, bottom_indices = self._quantize_with_codebook(bottom_context, self.bottom_codebook)
        codebook_indices = torch.stack([top_indices, bottom_indices], dim=-1)
        self._maybe_reset_dead_codes(encoded=encoded, codebook_indices=codebook_indices, is_last_train_step=is_last_train_step)
        top_latents = top_encoded + (top_quantized - top_encoded).detach()
        bottom_latents = bottom_context + (bottom_quantized - bottom_context).detach()
        latents = torch.cat([top_latents, bottom_latents], dim=-1)
        reconstruction = self.decode(latents)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        top_commitment_loss, bottom_commitment_loss = self.compute_commitment_loss_components(
            encoded, top_quantized, top_encoded, bottom_quantized, bottom_context
        )
        commitment_loss = top_commitment_loss + bottom_commitment_loss
        if self.training and self.config.use_ema_codebook:
            self.on_quantizer_training_step(encoded.detach(), codebook_indices.detach())
            top_codebook_loss = torch.zeros_like(commitment_loss)
            bottom_codebook_loss = torch.zeros_like(commitment_loss)
            codebook_loss = torch.zeros_like(commitment_loss)
            loss = reconstruction_loss + self.config.commitment_weight * commitment_loss
        else:
            top_codebook_loss = F.mse_loss(top_quantized, top_encoded.detach())
            bottom_codebook_loss = F.mse_loss(bottom_quantized, bottom_context.detach())
            codebook_loss = top_codebook_loss + bottom_codebook_loss
            loss = self.compute_total_loss(reconstruction_loss, commitment_loss, codebook_loss)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        if not use_return_dict:
            return loss, reconstruction, latents
        return HierarchicalQuantizedAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            quantized_latents=torch.cat([top_quantized, bottom_quantized], dim=-1),
            codebook_indices=codebook_indices,
            reconstruction_loss=reconstruction_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            top_quantized_latents=top_quantized,
            bottom_quantized_latents=bottom_quantized,
            top_codebook_indices=top_indices,
            bottom_codebook_indices=bottom_indices,
            top_commitment_loss=top_commitment_loss,
            bottom_commitment_loss=bottom_commitment_loss,
            top_codebook_loss=top_codebook_loss,
            bottom_codebook_loss=bottom_codebook_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "commitment_loss": commitment_loss,
                "codebook_loss": codebook_loss,
                "top_commitment_loss": top_commitment_loss,
                "bottom_commitment_loss": bottom_commitment_loss,
                "top_codebook_loss": top_codebook_loss,
                "bottom_codebook_loss": bottom_codebook_loss,
            },
        )
