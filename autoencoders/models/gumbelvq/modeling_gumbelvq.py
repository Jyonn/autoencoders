"""PyTorch implementation of a Gumbel-VQ autoencoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...data.base import TensorSpec
from ...modeling_outputs import GumbelQuantizedAutoencoderOutput
from ..base.modeling_vq import BaseVectorQuantizedAutoencoderModel
from .configuration_gumbelvq import GumbelQuantizedAutoencoderConfig


class GumbelQuantizedAutoencoderModel(BaseVectorQuantizedAutoencoderModel):
    """A vector quantized autoencoder using Gumbel-softmax assignments."""

    config_class = GumbelQuantizedAutoencoderConfig
    config: GumbelQuantizedAutoencoderConfig

    def __init__(self, **kwargs: object) -> None:
        config = kwargs.get("config")
        if kwargs.get("sample_spec") is None and config is not None and getattr(config, "input_dim", None) is not None:
            kwargs["sample_spec"] = TensorSpec(shape=(None, int(config.input_dim)))
        super().__init__(**kwargs)
        self.codebook = nn.Embedding(self.config.codebook_size, self.config.latent_dim)
        self._reset_codebook()

    def validate_core_spec(self) -> None:
        core_spec = self.core_spec
        if not isinstance(core_spec, TensorSpec):
            raise ValueError(f"{self.__class__.__name__} requires the core space to expose a TensorSpec.")
        if len(core_spec.shape) < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires the core TensorSpec to have rank >= 2, got {core_spec.shape}."
            )
        if core_spec.shape[-1] is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a concrete final feature dimension in the core TensorSpec."
            )

    def iter_codebook_index_sets(self, codebook_indices: torch.Tensor) -> list[torch.Tensor]:
        if codebook_indices.ndim == 2:
            return [codebook_indices.reshape(-1)]
        return super().iter_codebook_index_sets(codebook_indices)

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
        logits = -distances
        assignments = F.gumbel_softmax(
            logits,
            tau=self.config.temperature,
            hard=self.config.straight_through,
            dim=-1,
        )
        quantized_latents = assignments @ self.codebook.weight
        codebook_indices = assignments.argmax(dim=-1)
        return quantized_latents, codebook_indices

    def compute_assignment_entropy(self, encoded: torch.Tensor) -> torch.Tensor:
        distances = (
            encoded.pow(2).sum(dim=-1, keepdim=True)
            - 2 * encoded @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1)
        )
        logits = -distances / self.config.temperature
        probabilities = logits.softmax(dim=-1)
        return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def compute_codebook_loss(self, encoded: torch.Tensor, quantized_latents: torch.Tensor) -> torch.Tensor:
        del encoded, quantized_latents
        return self.codebook.weight.new_zeros(())

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        dead_code_mask = dead_code_mask.to(device=self.codebook.weight.device, dtype=torch.bool)
        dead_count = int(dead_code_mask.sum().item())
        if dead_count == 0:
            return 0
        if reference_latents is not None and reference_latents.numel() > 0:
            reference_latents = reference_latents.detach().to(self.codebook.weight.device).reshape(-1, self.config.latent_dim)
            sample_indices = torch.randint(0, reference_latents.shape[0], (dead_count,), device=self.codebook.weight.device)
            replacements = reference_latents[sample_indices]
        else:
            replacements = torch.empty(dead_count, self.config.latent_dim, device=self.codebook.weight.device)
            replacements.uniform_(-1.0 / self.config.codebook_size, 1.0 / self.config.codebook_size)
        self.codebook.weight.data[dead_code_mask] = replacements
        return dead_count

    def get_quantized_export_extras(self) -> dict[str, object]:
        extras = super().get_quantized_export_extras()
        extras["temperature"] = self.config.temperature
        extras["straight_through"] = self.config.straight_through
        extras["codebook"] = self.codebook.weight.detach().clone()
        return extras

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        is_last_train_step: bool | None = None,
        **kwargs: object,
    ) -> GumbelQuantizedAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        outputs = super().forward(
            inputs=inputs,
            return_dict=True,
            is_last_train_step=is_last_train_step,
            **kwargs,
        )
        assignment_entropy = self.compute_assignment_entropy(self.project_to_core(outputs.encoded))
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        if not use_return_dict:
            return outputs.loss, outputs.reconstruction, outputs.latents

        loss_dict = dict(outputs.loss_dict)
        loss_dict["assignment_entropy"] = assignment_entropy
        loss_dict["temperature"] = outputs.loss.new_tensor(float(self.config.temperature))
        return GumbelQuantizedAutoencoderOutput(
            loss=outputs.loss,
            reconstruction=outputs.reconstruction,
            latents=outputs.latents,
            encoded=outputs.encoded,
            quantized_latents=outputs.quantized_latents,
            codebook_indices=outputs.codebook_indices,
            reconstruction_loss=outputs.reconstruction_loss,
            commitment_loss=outputs.commitment_loss,
            codebook_loss=outputs.codebook_loss,
            assignment_entropy=assignment_entropy,
            temperature=outputs.loss.new_tensor(float(self.config.temperature)),
            loss_dict=loss_dict,
        )
