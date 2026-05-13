"""Base PyTorch model for autoencoder-family implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

from ...modeling_outputs import AutoencoderOutput
from ...modeling_utils import PreTrainedAutoencoderModel
from .configuration_base import BaseAutoencoderConfig


class BaseAutoencoderModel(PreTrainedAutoencoderModel, ABC):
    """Shared model skeleton for deterministic autoencoders."""

    config_class = BaseAutoencoderConfig

    def __init__(self, config: BaseAutoencoderConfig) -> None:
        super().__init__(config)

    @abstractmethod
    def encode(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode features into latent representations."""

    def latent_transform(self, encoded: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Hook for subclasses such as VAE or VQ-VAE."""
        return encoded

    @abstractmethod
    def decode(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Decode latent representations back into feature space."""

    def reconstruct(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        outputs = self.forward(inputs=inputs, return_dict=True, **kwargs)
        return outputs.reconstruction

    def compute_loss(self, reconstruction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.reconstruction_loss == "mse":
            return F.mse_loss(reconstruction, targets)
        if self.config.reconstruction_loss == "l1":
            return F.l1_loss(reconstruction, targets)
        raise ValueError(f"Unsupported reconstruction loss: {self.config.reconstruction_loss}")

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> AutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        model_inputs = self._resolve_inputs(inputs=inputs, features=features)
        encoded = self.encode(model_inputs, **kwargs)
        latents = self.latent_transform(encoded, **kwargs)
        reconstruction = self.decode(latents, **kwargs)

        loss_targets = model_inputs if targets is None else targets
        loss = self.compute_loss(reconstruction, loss_targets)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return AutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            hidden_states={"inputs": model_inputs},
            loss_dict={"reconstruction_loss": loss},
        )

    @staticmethod
    def _resolve_inputs(
        inputs: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs is not None and features is not None:
            raise ValueError("Provide only one of `inputs` or `features`.")
        if inputs is None and features is None:
            raise ValueError("One of `inputs` or `features` must be provided.")
        return inputs if inputs is not None else features

