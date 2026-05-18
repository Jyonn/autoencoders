"""Base model shared by vector-quantized autoencoder families."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn.functional as F

from ...function import center_distances_for_sinkhorn, sinkhorn_assignment_weights
from ...modeling_outputs import AutoencoderExport, BaseAutoencoderOutput, QuantizedAutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_vq import BaseVectorQuantizedAutoencoderConfig


class BaseVectorQuantizedAutoencoderModel(AutoencoderModel):
    """Shared VQ forward path, losses, and export hooks."""

    config_class = BaseVectorQuantizedAutoencoderConfig
    config: BaseVectorQuantizedAutoencoderConfig

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.register_buffer("_code_usage_counts", torch.zeros(0, dtype=torch.long), persistent=False)
        self._reference_latent_batches: list[torch.Tensor] = []
        self._last_dead_code_reset_count = 0
        self.register_buffer(
            "_codebooks_initialized_flag",
            torch.tensor(not self.config.kmeans_init, dtype=torch.bool),
        )

    @abstractmethod
    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map encoded latents to quantized latents and discrete indices."""

    def on_quantizer_training_step(self, encoded: torch.Tensor, codebook_indices: torch.Tensor) -> None:
        """Optional training-time hook such as EMA codebook updates."""

    def reset_dead_codes(self, dead_code_mask: torch.Tensor, reference_latents: torch.Tensor | None = None) -> int:
        return 0

    def initialize_codebooks(self, encoded: torch.Tensor) -> None:
        """Optional one-time initialization hook for learned codebooks."""

    def maybe_initialize_codebooks(self, encoded: torch.Tensor) -> None:
        if self.codebooks_initialized or not self.training:
            return
        self.initialize_codebooks(encoded.detach())
        self._codebooks_initialized_flag.fill_(True)

    @property
    def codebooks_initialized(self) -> bool:
        return bool(self._codebooks_initialized_flag.item())

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

    def assign_codebook_indices(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert final-axis distances into discrete codebook indices."""

        return self.assign_codebook_indices_for_slot(distances, slot=0)

    def get_sinkhorn_epsilon_for_slot(self, slot: int) -> float:
        epsilon = self.config.sinkhorn_epsilon
        if isinstance(epsilon, list):
            if len(epsilon) == 1:
                return float(epsilon[0])
            return float(epsilon[slot])
        return float(epsilon)

    def assign_codebook_indices_for_slot(self, distances: torch.Tensor, slot: int) -> torch.Tensor:
        """Convert final-axis distances into discrete codebook indices for one codebook slot."""

        flattened_distances = distances.reshape(-1, distances.shape[-1])
        sinkhorn_epsilon = self.get_sinkhorn_epsilon_for_slot(slot)
        if self.config.assignment_strategy == "nearest" or sinkhorn_epsilon <= 0:
            indices = flattened_distances.argmin(dim=-1)
        else:
            centered_distances = center_distances_for_sinkhorn(flattened_distances).double()
            assignment_weights = sinkhorn_assignment_weights(
                centered_distances,
                epsilon=sinkhorn_epsilon,
                num_iters=self.config.sinkhorn_iters,
            )
            if torch.isnan(assignment_weights).any() or torch.isinf(assignment_weights).any():
                raise ValueError("Sinkhorn assignment produced NaN or Inf values.")
            indices = assignment_weights.argmax(dim=-1).to(device=flattened_distances.device)
        return indices.reshape(distances.shape[:-1])

    def get_quantized_export_extras(self) -> dict[str, object]:
        return {
            "codebook_size": self.config.codebook_size,
            "assignment_strategy": self.config.assignment_strategy,
            "sinkhorn_epsilon": self.config.sinkhorn_epsilon,
            "sinkhorn_iters": self.config.sinkhorn_iters,
            "kmeans_init": self.config.kmeans_init,
            "kmeans_iters": self.config.kmeans_iters,
            "use_ema_codebook": self.config.use_ema_codebook,
            "dead_code_reset": self.config.dead_code_reset,
            "dead_code_threshold": self.config.dead_code_threshold,
        }

    def consume_dead_code_reset_count(self) -> int:
        dead_code_reset_count = self._last_dead_code_reset_count
        self._last_dead_code_reset_count = 0
        return dead_code_reset_count

    def iter_codebook_index_sets(self, codebook_indices: torch.Tensor) -> list[torch.Tensor]:
        if codebook_indices.ndim == 1:
            return [codebook_indices.reshape(-1)]
        if codebook_indices.ndim == 2:
            return [codebook_indices[:, codebook_index].reshape(-1) for codebook_index in range(codebook_indices.shape[1])]
        if codebook_indices.ndim == 3:
            return [codebook_indices[..., codebook_index].reshape(-1) for codebook_index in range(codebook_indices.shape[-1])]
        raise ValueError("Quantized models expect codebook_indices with rank 1, 2, or 3.")

    def _prepare_code_usage_counts(self, codebook_indices: torch.Tensor) -> None:
        index_sets = self.iter_codebook_index_sets(codebook_indices)
        if len(index_sets) == 1:
            expected_shape = (self.config.codebook_size,)
        else:
            expected_shape = (len(index_sets), self.config.codebook_size)

        if tuple(self._code_usage_counts.shape) != expected_shape or self._code_usage_counts.device != codebook_indices.device:
            self._code_usage_counts = torch.zeros(expected_shape, dtype=torch.long, device=codebook_indices.device)

    def _accumulate_code_usage(self, codebook_indices: torch.Tensor) -> None:
        self._prepare_code_usage_counts(codebook_indices)
        index_sets = self.iter_codebook_index_sets(codebook_indices)
        if len(index_sets) == 1:
            self._code_usage_counts += torch.bincount(
                index_sets[0],
                minlength=self.config.codebook_size,
            )
            return

        for codebook_index, flattened_indices in enumerate(index_sets):
            self._code_usage_counts[codebook_index] += torch.bincount(
                flattened_indices,
                minlength=self.config.codebook_size,
            )

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
        self._reference_latent_batches.append(encoded.detach().reshape(-1, encoded.shape[-1]))
        if not is_last_train_step:
            return

        dead_code_mask = self._code_usage_counts <= self.config.dead_code_threshold
        reference_latents = (
            torch.cat(self._reference_latent_batches, dim=0)
            if self._reference_latent_batches
            else None
        )
        self._last_dead_code_reset_count = int(self.reset_dead_codes(dead_code_mask, reference_latents))
        self._code_usage_counts.zero_()
        self._reference_latent_batches.clear()

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        is_last_train_step: bool | None = None,
        **kwargs: object,
    ) -> QuantizedAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        core_inputs = self.project_to_core(encoded)
        self.maybe_initialize_codebooks(core_inputs)
        quantized_latents, codebook_indices = self.quantize(core_inputs)
        self._maybe_reset_dead_codes(
            encoded=core_inputs,
            codebook_indices=codebook_indices,
            is_last_train_step=is_last_train_step,
        )
        latents = core_inputs + (quantized_latents - core_inputs).detach()
        decoder_inputs = self.project_from_core(latents)
        reconstruction = self.decode(decoder_inputs)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        commitment_loss = self.compute_commitment_loss(core_inputs, quantized_latents)
        if self.training and self.config.use_ema_codebook:
            self.on_quantizer_training_step(core_inputs.detach(), codebook_indices.detach())
            codebook_loss = torch.zeros_like(commitment_loss)
            loss = reconstruction_loss + self.config.commitment_weight * commitment_loss
        else:
            codebook_loss = self.compute_codebook_loss(core_inputs, quantized_latents)
            loss = self.compute_total_loss(reconstruction_loss, commitment_loss, codebook_loss)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return QuantizedAutoencoderOutput(
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
        outputs: BaseAutoencoderOutput,
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
