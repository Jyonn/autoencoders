"""PyTorch implementation of a vector-quantized autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import AutoencoderExport, AutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_vqvae import VectorQuantizedAutoencoderConfig


class VectorQuantizedAutoencoderModel(AutoencoderModel):
    """A simple MLP VQ-VAE for vector-like feature inputs."""

    config_class = VectorQuantizedAutoencoderConfig

    def __init__(self, config: VectorQuantizedAutoencoderConfig) -> None:
        super().__init__(config)
        self.codebook = nn.Embedding(self.config.codebook_size, self.config.latent_dim)
        self._reset_codebook()

    def _reset_codebook(self) -> None:
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / self.config.codebook_size,
            1.0 / self.config.codebook_size,
        )

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = (
            encoded.pow(2).sum(dim=-1, keepdim=True)
            - 2 * encoded @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1)
        )
        codebook_indices = distances.argmin(dim=-1)
        quantized_latents = self.codebook(codebook_indices)
        return quantized_latents, codebook_indices

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
    ) -> AutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        quantized_latents, codebook_indices = self.quantize(encoded)
        latents = encoded + (quantized_latents - encoded).detach()
        reconstruction = self.decode(latents)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        commitment_loss = F.mse_loss(encoded, quantized_latents.detach())
        codebook_loss = F.mse_loss(quantized_latents, encoded.detach())
        loss = (
            reconstruction_loss
            + self.config.commitment_weight * commitment_loss
            + self.config.codebook_weight * codebook_loss
        )
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return AutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            quantized_latents=quantized_latents,
            codebook_indices=codebook_indices,
            reconstruction_loss=reconstruction_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "commitment_loss": commitment_loss,
                "codebook_loss": codebook_loss,
            },
        )

    def _build_export(
        self,
        *,
        inputs: torch.Tensor,
        outputs: AutoencoderOutput,
        include_reconstruction: bool,
        metadata: dict[str, object] | None,
    ) -> AutoencoderExport:
        artifact = super()._build_export(
            inputs=inputs,
            outputs=outputs,
            include_reconstruction=include_reconstruction,
            metadata=metadata,
        )
        artifact.quantized_latents = outputs.quantized_latents
        artifact.codebook_indices = outputs.codebook_indices
        artifact.extras["codebook_size"] = self.config.codebook_size
        artifact.extras["codebook"] = self.codebook.weight.detach().clone()
        return artifact
