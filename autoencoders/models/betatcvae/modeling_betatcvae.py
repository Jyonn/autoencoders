"""PyTorch implementation of a Beta-TCVAE."""

from __future__ import annotations

import math

import torch

from ...modeling_outputs import BetaTCVariationalAutoencoderOutput
from ..vae.modeling_vae import VariationalAutoencoderModel
from .configuration_betatcvae import BetaTCVariationalAutoencoderConfig


class BetaTCVariationalAutoencoderModel(VariationalAutoencoderModel):
    """A VAE with decomposed KL regularization."""

    config_class = BetaTCVariationalAutoencoderConfig
    config: BetaTCVariationalAutoencoderConfig

    def _log_density_gaussian(self, sample: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        normalization = math.log(2.0 * math.pi)
        return -0.5 * (((sample - mean).pow(2) * torch.exp(-logvar)) + logvar + normalization)

    def decompose_kl(
        self,
        latents: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_qzx = self._log_density_gaussian(latents, posterior_mean, posterior_logvar).sum(dim=-1)
        latent_expanded = latents.unsqueeze(1)
        mean_expanded = posterior_mean.unsqueeze(0)
        logvar_expanded = posterior_logvar.unsqueeze(0)
        log_qz_prob = self._log_density_gaussian(latent_expanded, mean_expanded, logvar_expanded)
        log_qz_product = torch.logsumexp(log_qz_prob, dim=1) - math.log(latents.shape[0])
        log_qz = torch.logsumexp(log_qz_prob.sum(dim=-1), dim=1) - math.log(latents.shape[0])
        log_pz = self._log_density_gaussian(
            latents,
            torch.zeros_like(latents),
            torch.zeros_like(latents),
        ).sum(dim=-1)

        mutual_information_loss = (log_qzx - log_qz).mean()
        total_correlation_loss = (log_qz - log_qz_product.sum(dim=-1)).mean()
        dimension_wise_kl_loss = (log_qz_product.sum(dim=-1) - log_pz).mean()
        return mutual_information_loss, total_correlation_loss, dimension_wise_kl_loss

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> BetaTCVariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        outputs = super().forward(
            inputs=inputs,
            return_dict=True,
            sample_posterior=sample_posterior,
            global_step=global_step,
            current_epoch=current_epoch,
        )
        mutual_information_loss, total_correlation_loss, dimension_wise_kl_loss = self.decompose_kl(
            outputs.latents,
            outputs.posterior_mean,
            outputs.posterior_logvar,
        )
        loss = (
            outputs.reconstruction_loss
            + self.config.mutual_information_weight * mutual_information_loss
            + self.config.total_correlation_weight * total_correlation_loss
            + self.config.dimension_wise_kl_weight * dimension_wise_kl_loss
        )
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        if not use_return_dict:
            return loss, outputs.reconstruction, outputs.latents

        loss_dict = dict(outputs.loss_dict)
        loss_dict.update(
            loss=loss,
            mutual_information_loss=mutual_information_loss,
            total_correlation_loss=total_correlation_loss,
            dimension_wise_kl_loss=dimension_wise_kl_loss,
        )
        return BetaTCVariationalAutoencoderOutput(
            loss=loss,
            reconstruction=outputs.reconstruction,
            latents=outputs.latents,
            encoded=outputs.encoded,
            posterior_mean=outputs.posterior_mean,
            posterior_logvar=outputs.posterior_logvar,
            reconstruction_loss=outputs.reconstruction_loss,
            kl_loss=outputs.kl_loss,
            free_bits_kl_loss=outputs.free_bits_kl_loss,
            effective_kl_weight=outputs.effective_kl_weight,
            mutual_information_loss=mutual_information_loss,
            total_correlation_loss=total_correlation_loss,
            dimension_wise_kl_loss=dimension_wise_kl_loss,
            loss_dict=loss_dict,
        )
