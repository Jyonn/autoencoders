"""PyTorch implementation of a denoising autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import DenoisingAutoencoderOutput
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
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        add_noise: bool | None = None,
        corrupted_inputs: torch.Tensor | None = None,
        **kwargs: object,
    ) -> DenoisingAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        apply_noise = self.training if add_noise is None else add_noise
        if not self.training and add_noise is None:
            apply_noise = self.config.apply_noise_in_eval

        if corrupted_inputs is not None:
            model_inputs = corrupted_inputs
        elif apply_noise:
            model_inputs = self.corrupt_inputs(inputs)
        else:
            model_inputs = inputs

        encoded = self.encode(model_inputs)
        latents = self.latent_transform(encoded)
        reconstruction = self.decode(latents)

        loss = self.compute_loss(reconstruction, inputs)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return DenoisingAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            hidden_states={
                "inputs": inputs,
                "corrupted_inputs": model_inputs,
            },
            loss_dict={"reconstruction_loss": loss},
        )
