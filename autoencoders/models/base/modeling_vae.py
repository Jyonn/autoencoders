"""Base model shared by variational autoencoder families."""

from __future__ import annotations

from abc import abstractmethod

import torch

from ...modeling_outputs import VariationalAutoencoderOutput
from .configuration_vae import BaseVariationalAutoencoderConfig
from .modeling_base import BaseAutoencoderModel


class BaseVariationalAutoencoderModel(BaseAutoencoderModel):
    """Shared VAE forward path and KL utilities."""

    config_class = BaseVariationalAutoencoderConfig

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs into posterior parameters."""

    def reparameterize(self, posterior_mean: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * posterior_logvar)
        epsilon = torch.randn_like(std)
        return posterior_mean + epsilon * std

    def sample_latents(
        self,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        sample_posterior: bool | None = None,
    ) -> torch.Tensor:
        if sample_posterior is None:
            sample_posterior = self.training or not self.config.use_mean_in_eval
        if sample_posterior:
            return self.reparameterize(posterior_mean, posterior_logvar)
        return posterior_mean

    def compute_kl_loss(self, posterior_mean: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        kl_per_example = -0.5 * torch.sum(
            1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp(),
            dim=-1,
        )
        return kl_per_example.mean()

    def compute_total_loss(
        self,
        reconstruction_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        *,
        kl_weight: float | None = None,
    ) -> torch.Tensor:
        effective_kl_weight = self.config.kl_weight if kl_weight is None else kl_weight
        return reconstruction_loss + effective_kl_weight * kl_loss

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
    ) -> VariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        posterior_mean, posterior_logvar = self.encode(inputs)
        latents = self.sample_latents(
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            sample_posterior=sample_posterior,
        )
        reconstruction = self.decode(latents)
        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        kl_loss = self.compute_kl_loss(posterior_mean, posterior_logvar)
        loss = self.compute_total_loss(reconstruction_loss, kl_loss)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return VariationalAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=posterior_mean,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            },
        )
