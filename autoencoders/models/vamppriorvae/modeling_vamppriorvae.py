"""PyTorch implementation of a variational autoencoder with a learned VampPrior."""

from __future__ import annotations

import math

import torch
from torch import nn

from ...modeling_outputs import VampPriorVariationalAutoencoderOutput
from ..vae.modeling_vae import VariationalAutoencoderModel
from .configuration_vamppriorvae import VampPriorVariationalAutoencoderConfig


class VampPriorVariationalAutoencoderModel(VariationalAutoencoderModel):
    """A VAE whose prior is a learned mixture induced by pseudo-inputs."""

    config_class = VampPriorVariationalAutoencoderConfig

    def __init__(
        self,
        config: VampPriorVariationalAutoencoderConfig,
        **kwargs: object,
    ) -> None:
        super().__init__(config, **kwargs)
        pseudo_inputs = torch.randn(config.num_pseudo_inputs, config.input_dim) * config.pseudo_input_std
        self.pseudo_inputs = nn.Parameter(pseudo_inputs)

    @staticmethod
    def _gaussian_log_prob(samples: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (samples - mean).pow(2) / logvar.exp()
        ).sum(dim=-1)

    def get_prior_posterior_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(self.pseudo_inputs)

    def compute_kl_components(
        self,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_qzx = self._gaussian_log_prob(latents, posterior_mean, posterior_logvar)
        pseudo_mean, pseudo_logvar = self.get_prior_posterior_parameters()
        expanded_latents = latents.unsqueeze(1)
        component_log_probs = -0.5 * (
            math.log(2.0 * math.pi)
            + pseudo_logvar.unsqueeze(0)
            + (expanded_latents - pseudo_mean.unsqueeze(0)).pow(2) / pseudo_logvar.unsqueeze(0).exp()
        ).sum(dim=-1)
        log_pz = torch.logsumexp(component_log_probs, dim=1) - math.log(self.config.num_pseudo_inputs)
        kl_per_example = log_qzx - log_pz
        return kl_per_example, pseudo_mean, pseudo_logvar

    def compute_free_bits_kl_loss_from_examples(self, kl_per_example: torch.Tensor) -> torch.Tensor:
        mean_kl = kl_per_example.mean()
        minimum_kl = float(self.config.free_bits) * float(self.config.latent_dim)
        return torch.clamp(mean_kl, min=minimum_kl)

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> VampPriorVariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        posterior_mean, posterior_logvar = self.encode(inputs)
        latents = self.sample_latents(
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            sample_posterior=sample_posterior,
        )
        reconstruction = self.decode(latents)
        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        kl_per_example, pseudo_mean, pseudo_logvar = self.compute_kl_components(
            posterior_mean,
            posterior_logvar,
            latents,
        )
        kl_loss = kl_per_example.mean()
        free_bits_kl_loss = self.compute_free_bits_kl_loss_from_examples(kl_per_example)
        effective_kl_weight = self.get_current_kl_weight(global_step=global_step, current_epoch=current_epoch)
        loss = self.compute_total_loss(reconstruction_loss, free_bits_kl_loss, kl_weight=effective_kl_weight)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return VampPriorVariationalAutoencoderOutput(
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
            pseudo_posterior_mean=pseudo_mean,
            pseudo_posterior_logvar=pseudo_logvar,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "free_bits_kl_loss": free_bits_kl_loss,
                "effective_kl_weight": loss.new_tensor(effective_kl_weight),
            },
        )
