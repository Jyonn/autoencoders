"""PyTorch implementation of a residual finite-scalar quantized autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...modeling_outputs import ResidualFiniteScalarQuantizedAutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_rfsq import ResidualFiniteScalarQuantizedAutoencoderConfig


class ResidualFiniteScalarQuantizedAutoencoderModel(AutoencoderModel):
    """An autoencoder with stacked residual scalar quantizers."""

    config_class = ResidualFiniteScalarQuantizedAutoencoderConfig
    config: ResidualFiniteScalarQuantizedAutoencoderConfig

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        levels = torch.linspace(
            -self.config.quantization_bound,
            self.config.quantization_bound,
            steps=self.config.num_levels,
        )
        self.register_buffer("levels", levels)

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = encoded
        quantized_sum = torch.zeros_like(encoded)
        index_steps: list[torch.Tensor] = []
        for _ in range(self.config.num_quantizers):
            distances = (residual.unsqueeze(-1) - self.levels.view(1, 1, -1)).abs()
            indices = distances.argmin(dim=-1)
            quantized = self.levels[indices]
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            index_steps.append(indices)
        codebook_indices = torch.stack(index_steps, dim=1).reshape(encoded.shape[0], -1)
        return quantized_sum, codebook_indices

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> ResidualFiniteScalarQuantizedAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        core_inputs = self.project_to_core(encoded)
        quantized_latents, codebook_indices = self.quantize(core_inputs)
        latents = core_inputs + (quantized_latents - core_inputs).detach()
        reconstruction = self.decode(self.project_from_core(latents))
        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        commitment_loss = F.mse_loss(core_inputs, quantized_latents.detach())
        quantization_residual_loss = F.mse_loss(
            core_inputs - quantized_latents.detach(),
            torch.zeros_like(core_inputs),
        )
        codebook_loss = torch.zeros_like(commitment_loss)
        loss = reconstruction_loss + self.config.commitment_weight * commitment_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        if not use_return_dict:
            return loss, reconstruction, latents
        return ResidualFiniteScalarQuantizedAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            quantized_latents=quantized_latents,
            codebook_indices=codebook_indices,
            reconstruction_loss=reconstruction_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            quantization_residual_loss=quantization_residual_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "commitment_loss": commitment_loss,
                "codebook_loss": codebook_loss,
                "quantization_residual_loss": quantization_residual_loss,
            },
        )

    def export(self, *args, **kwargs):
        return super().export(*args, **kwargs)

    def _build_export(self, *, inputs, outputs, include_reconstruction, metadata):
        artifact = super()._build_export(
            inputs=inputs,
            outputs=outputs,
            include_reconstruction=include_reconstruction,
            metadata=metadata,
        )
        artifact.quantized_latents = outputs.quantized_latents
        artifact.codebook_indices = outputs.codebook_indices
        artifact.extras["num_levels"] = self.config.num_levels
        artifact.extras["num_quantizers"] = self.config.num_quantizers
        artifact.extras["levels"] = self.levels.detach().clone()
        return artifact
