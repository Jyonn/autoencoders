"""PyTorch implementation of a finite scalar quantized autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...modeling_outputs import AutoencoderExport, AutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_fsq import FiniteScalarQuantizedAutoencoderConfig


class FiniteScalarQuantizedAutoencoderModel(AutoencoderModel):
    """An autoencoder with per-dimension finite scalar quantization."""

    config_class = FiniteScalarQuantizedAutoencoderConfig

    def __init__(self, config: FiniteScalarQuantizedAutoencoderConfig) -> None:
        super().__init__(config)
        levels = torch.linspace(
            -self.config.quantization_bound,
            self.config.quantization_bound,
            steps=self.config.num_levels,
        )
        self.register_buffer("levels", levels)

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = (encoded.unsqueeze(-1) - self.levels.view(1, 1, -1)).abs()
        codebook_indices = distances.argmin(dim=-1)
        quantized_latents = self.levels[codebook_indices]
        return quantized_latents, codebook_indices

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        return 0

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
        loss = reconstruction_loss + self.config.commitment_weight * commitment_loss
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
            codebook_loss=torch.zeros_like(commitment_loss),
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "commitment_loss": commitment_loss,
                "codebook_loss": torch.zeros_like(commitment_loss),
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
        artifact.extras["num_levels"] = self.config.num_levels
        artifact.extras["levels"] = self.levels.detach().clone()
        return artifact
