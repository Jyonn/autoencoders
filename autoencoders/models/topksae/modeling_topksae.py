"""PyTorch implementation of a top-k sparse autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import TopKSparseAutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_topksae import TopKSparseAutoencoderConfig


class TopKSparseAutoencoderModel(AutoencoderModel):
    """An autoencoder that keeps only the top-k latent activations per sample."""

    config_class = TopKSparseAutoencoderConfig

    def apply_topk(self, latents: torch.Tensor) -> torch.Tensor:
        topk_values, topk_indices = torch.topk(latents.abs(), k=self.config.topk, dim=-1)
        mask = torch.zeros_like(latents, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        return latents * mask.to(latents.dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> TopKSparseAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        latents = self.apply_topk(self.latent_transform(encoded))
        reconstruction = self.decode(latents)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        topk_sparsity = (latents != 0).to(latents.dtype).mean()
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return reconstruction_loss, reconstruction, latents

        return TopKSparseAutoencoderOutput(
            loss=reconstruction_loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            reconstruction_loss=reconstruction_loss,
            topk_sparsity=topk_sparsity,
            loss_dict={
                "loss": reconstruction_loss,
                "reconstruction_loss": reconstruction_loss,
                "topk_sparsity": topk_sparsity,
            },
        )
