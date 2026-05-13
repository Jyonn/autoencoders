"""PyTorch implementation of a denoising autoencoder."""

from __future__ import annotations

from typing import Any

import torch

from ...modeling_outputs import AutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_dae import DenoisingAutoencoderConfig


class DenoisingAutoencoderModel(AutoencoderModel):
    """A deterministic autoencoder trained to reconstruct clean inputs from noisy ones."""

    config_class = DenoisingAutoencoderConfig

    def __init__(self, config: DenoisingAutoencoderConfig) -> None:
        super().__init__(config)

    def corrupt_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.config.noise_type == "gaussian":
            if self.config.noise_std == 0:
                return inputs.clone()
            noise = torch.randn_like(inputs) * self.config.noise_std
            return inputs + noise

        if self.config.noise_type == "masking":
            if self.config.masking_ratio == 0:
                return inputs.clone()
            keep_mask = torch.rand_like(inputs) > self.config.masking_ratio
            return inputs * keep_mask.to(inputs.dtype)

        raise ValueError(f"Unsupported noise_type: {self.config.noise_type}")

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_dict: bool | None = None,
        add_noise: bool | None = None,
        corrupted_inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        clean_inputs = self._resolve_inputs(inputs=inputs, features=features)
        apply_noise = self.training if add_noise is None else add_noise
        if not self.training and add_noise is None:
            apply_noise = self.config.apply_noise_in_eval

        if corrupted_inputs is not None:
            model_inputs = corrupted_inputs
        elif apply_noise:
            model_inputs = self.corrupt_inputs(clean_inputs)
        else:
            model_inputs = clean_inputs

        encoded = self.encode(model_inputs, **kwargs)
        latents = self.latent_transform(encoded, **kwargs)
        reconstruction = self.decode(latents, **kwargs)

        loss_targets = clean_inputs if targets is None else targets
        loss = self.compute_loss(reconstruction, loss_targets)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return AutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            hidden_states={
                "inputs": clean_inputs,
                "corrupted_inputs": model_inputs,
            },
            loss_dict={"reconstruction_loss": loss},
        )
