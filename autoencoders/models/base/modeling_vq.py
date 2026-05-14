"""Base model shared by vector-quantized autoencoder families."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn.functional as F

from ...modeling_outputs import AutoencoderExport, AutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_vq import BaseVectorQuantizedAutoencoderConfig


class BaseVectorQuantizedAutoencoderModel(AutoencoderModel):
    """Shared VQ forward path, losses, and export hooks."""

    config_class = BaseVectorQuantizedAutoencoderConfig

    @abstractmethod
    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map encoded latents to quantized latents and discrete indices."""

    def on_quantizer_training_step(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        """Optional training-time hook such as EMA codebook updates."""

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        return 0

    def compute_commitment_loss(self, encoded: torch.Tensor, quantized_latents: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(encoded, quantized_latents.detach())

    def compute_codebook_loss(self, encoded: torch.Tensor, quantized_latents: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(quantized_latents, encoded.detach())

    def compute_total_loss(
        self,
        reconstruction_loss: torch.Tensor,
        commitment_loss: torch.Tensor,
        codebook_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            reconstruction_loss
            + self.config.commitment_weight * commitment_loss
            + self.config.codebook_weight * codebook_loss
        )

    def get_quantized_export_extras(self) -> dict[str, object]:
        return {
            "codebook_size": self.config.codebook_size,
            "use_ema_codebook": self.config.use_ema_codebook,
        }

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
        commitment_loss = self.compute_commitment_loss(encoded, quantized_latents)
        if self.training and self.config.use_ema_codebook:
            self.on_quantizer_training_step(encoded.detach(), codebook_indices.detach())
            codebook_loss = torch.zeros_like(commitment_loss)
            loss = reconstruction_loss + self.config.commitment_weight * commitment_loss
        else:
            codebook_loss = self.compute_codebook_loss(encoded, quantized_latents)
            loss = self.compute_total_loss(reconstruction_loss, commitment_loss, codebook_loss)
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
        artifact.extras.update(self.get_quantized_export_extras())
        return artifact
