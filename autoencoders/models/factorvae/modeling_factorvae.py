"""PyTorch implementation of a FactorVAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import FactorVariationalAutoencoderOutput
from ..vae.modeling_vae import VariationalAutoencoderModel
from .configuration_factorvae import FactorVariationalAutoencoderConfig


class FactorVariationalAutoencoderModel(VariationalAutoencoderModel):
    """A variational autoencoder with a latent total-correlation penalty."""

    config_class = FactorVariationalAutoencoderConfig

    def __init__(
        self,
        config: FactorVariationalAutoencoderConfig,
        **kwargs: object,
    ) -> None:
        super().__init__(config, **kwargs)
        self.discriminator = self._build_discriminator()

    def discriminate(self, latents: torch.Tensor) -> torch.Tensor:
        return self.discriminator(latents)

    def permute_dims(self, latents: torch.Tensor) -> torch.Tensor:
        permuted = []
        for latent_index in range(latents.shape[1]):
            permutation = torch.randperm(latents.shape[0], device=latents.device)
            permuted.append(latents[permutation, latent_index])
        return torch.stack(permuted, dim=1)

    def compute_total_correlation_loss(self, latents: torch.Tensor) -> torch.Tensor:
        logits = self.discriminate(latents)
        return (logits[:, 0] - logits[:, 1]).mean()

    def compute_discriminator_loss(self, latents: torch.Tensor, permuted_latents: torch.Tensor) -> torch.Tensor:
        real_logits = self.discriminate(latents)
        permuted_logits = self.discriminate(permuted_latents)
        real_targets = torch.zeros(latents.shape[0], dtype=torch.long, device=latents.device)
        permuted_targets = torch.ones(latents.shape[0], dtype=torch.long, device=latents.device)
        real_loss = F.cross_entropy(real_logits, real_targets)
        permuted_loss = F.cross_entropy(permuted_logits, permuted_targets)
        return 0.5 * (real_loss + permuted_loss)

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> FactorVariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        posterior_mean, posterior_logvar = self.encode(inputs)
        latents = self.sample_latents(
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            sample_posterior=sample_posterior,
        )
        reconstruction = self.decode(latents)
        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        kl_loss = self.compute_kl_loss(posterior_mean, posterior_logvar)
        free_bits_kl_loss = self.compute_free_bits_kl_loss(posterior_mean, posterior_logvar)
        total_correlation_loss = self.compute_total_correlation_loss(latents)
        effective_kl_weight = self.get_current_kl_weight(global_step=global_step, current_epoch=current_epoch)
        loss = (
            reconstruction_loss
            + effective_kl_weight * free_bits_kl_loss
            + self.config.tc_weight * total_correlation_loss
        )
        permuted_latents = self.permute_dims(latents.detach())
        discriminator_loss = self.compute_discriminator_loss(latents.detach(), permuted_latents)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return FactorVariationalAutoencoderOutput(
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
            total_correlation_loss=total_correlation_loss,
            discriminator_loss=discriminator_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "free_bits_kl_loss": free_bits_kl_loss,
                "effective_kl_weight": loss.new_tensor(effective_kl_weight),
                "total_correlation_loss": total_correlation_loss,
                "discriminator_loss": discriminator_loss,
            },
        )

    def _build_discriminator(self) -> nn.Sequential:
        dims = [self.config.latent_dim, *self.config.discriminator_hidden_dims, 2]
        layers: list[nn.Module] = []

        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            is_last_layer = index == len(dims) - 2
            if not is_last_layer:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)
