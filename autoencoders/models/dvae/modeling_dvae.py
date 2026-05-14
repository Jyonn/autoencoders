"""PyTorch implementation of a denoising variational autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import DenoisingVariationalAutoencoderOutput
from ..vae.modeling_vae import VariationalAutoencoderModel
from .configuration_dvae import DenoisingVariationalAutoencoderConfig


class DenoisingVariationalAutoencoderModel(VariationalAutoencoderModel):
    """A VAE trained to reconstruct clean inputs from noisy observations."""

    config_class = DenoisingVariationalAutoencoderConfig

    def __init__(self, config: DenoisingVariationalAutoencoderConfig) -> None:
        super().__init__(config)

    def corrupt_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.config.noise_type == "gaussian":
            if self.config.noise_std == 0:
                return inputs.clone()
            return inputs + torch.randn_like(inputs) * self.config.noise_std
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
        sample_posterior: bool | None = None,
        add_noise: bool | None = None,
        corrupted_inputs: torch.Tensor | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> DenoisingVariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        apply_noise = self.training if add_noise is None else add_noise
        if not self.training and add_noise is None:
            apply_noise = self.config.apply_noise_in_eval

        if corrupted_inputs is not None:
            model_inputs = corrupted_inputs
        elif apply_noise:
            model_inputs = self.corrupt_inputs(inputs)
        else:
            model_inputs = inputs

        posterior_mean, posterior_logvar = self.encode(model_inputs)
        latents = self.sample_latents(
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            sample_posterior=sample_posterior,
        )
        reconstruction = self.decode(latents)
        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        kl_loss = self.compute_kl_loss(posterior_mean, posterior_logvar)
        free_bits_kl_loss = self.compute_free_bits_kl_loss(posterior_mean, posterior_logvar)
        effective_kl_weight = self.get_current_kl_weight(global_step=global_step, current_epoch=current_epoch)
        loss = self.compute_total_loss(reconstruction_loss, free_bits_kl_loss, kl_weight=effective_kl_weight)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return DenoisingVariationalAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=posterior_mean,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            free_bits_kl_loss=free_bits_kl_loss,
            effective_kl_weight=effective_kl_weight,
            hidden_states={"inputs": inputs, "corrupted_inputs": model_inputs},
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "free_bits_kl_loss": free_bits_kl_loss,
                "effective_kl_weight": loss.new_tensor(effective_kl_weight),
            },
        )
