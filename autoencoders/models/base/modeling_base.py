"""Base PyTorch model for autoencoder-family implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from ...modeling_outputs import AutoencoderExport, BaseAutoencoderOutput
from ...modeling_utils import PreTrainedAutoencoderModel
from .configuration_base import BaseAutoencoderConfig


class BaseAutoencoderModel(PreTrainedAutoencoderModel, ABC):
    """Shared model skeleton for deterministic autoencoders."""

    config_class = BaseAutoencoderConfig
    requires_grad_in_eval = False

    def __init__(self, config: BaseAutoencoderConfig) -> None:
        super().__init__(config)

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into latent representations."""

    def latent_transform(self, encoded: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses such as VAE or VQ-VAE."""
        return encoded

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back into feature space."""

    def reconstruct(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(inputs=inputs, return_dict=True)
        return outputs.reconstruction

    def export(
        self,
        inputs: torch.Tensor,
        include_reconstruction: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> AutoencoderExport:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs=inputs, return_dict=True)
        if was_training:
            self.train()
        return self._build_export(
            inputs=inputs,
            outputs=outputs,
            include_reconstruction=include_reconstruction,
            metadata=metadata,
        )

    def compute_loss(self, reconstruction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.reconstruction_loss == "mse":
            return F.mse_loss(reconstruction, targets)
        if self.config.reconstruction_loss == "l1":
            return F.l1_loss(reconstruction, targets)
        raise ValueError(f"Unsupported reconstruction loss: {self.config.reconstruction_loss}")

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> BaseAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        latents = self.latent_transform(encoded)
        reconstruction = self.decode(latents)

        loss = self.compute_loss(reconstruction, inputs)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return BaseAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            loss_dict={"reconstruction_loss": loss},
        )

    def _build_export(
        self,
        *,
        inputs: torch.Tensor,
        outputs: BaseAutoencoderOutput,
        include_reconstruction: bool,
        metadata: dict[str, object] | None,
    ) -> AutoencoderExport:
        export_metadata: dict[str, object] = {
            "input_shape": list(inputs.shape),
            "latent_shape": list(outputs.latents.shape) if outputs.latents is not None else None,
        }
        if metadata is not None:
            export_metadata.update(metadata)

        return AutoencoderExport(
            model_type=self.config.model_type,
            latents=outputs.latents,
            reconstruction=outputs.reconstruction if include_reconstruction else None,
            encoded=outputs.encoded,
            posterior_mean=getattr(outputs, "posterior_mean", None),
            posterior_logvar=getattr(outputs, "posterior_logvar", None),
            quantized_latents=getattr(outputs, "quantized_latents", None),
            codebook_indices=getattr(outputs, "codebook_indices", None),
            metadata=export_metadata,
        )
