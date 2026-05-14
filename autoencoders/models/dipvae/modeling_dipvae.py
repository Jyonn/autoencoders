"""PyTorch implementation of a DIP-VAE."""

from __future__ import annotations

import torch

from ...modeling_outputs import DIPVariationalAutoencoderOutput
from ..vae.modeling_vae import VariationalAutoencoderModel
from .configuration_dipvae import DIPVariationalAutoencoderConfig


class DIPVariationalAutoencoderModel(VariationalAutoencoderModel):
    """A variational autoencoder with covariance regularization."""

    config_class = DIPVariationalAutoencoderConfig

    def compute_dip_loss(self, posterior_mean: torch.Tensor) -> torch.Tensor:
        centered_mean = posterior_mean - posterior_mean.mean(dim=0, keepdim=True)
        covariance = centered_mean.transpose(0, 1) @ centered_mean / max(posterior_mean.shape[0], 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        diagonal_error = torch.diagonal(covariance) - 1.0
        offdiag_loss = off_diagonal.pow(2).sum()
        diag_loss = diagonal_error.pow(2).sum()
        return (
            self.config.dip_offdiag_weight * offdiag_loss
            + self.config.dip_diag_weight * diag_loss
        )

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> DIPVariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        outputs = super().forward(
            inputs=inputs,
            return_dict=True,
            sample_posterior=sample_posterior,
            global_step=global_step,
            current_epoch=current_epoch,
        )
        dip_loss = self.compute_dip_loss(outputs.posterior_mean)
        loss = outputs.loss + self.config.dip_weight * dip_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, outputs.reconstruction, outputs.latents

        loss_dict = dict(outputs.loss_dict)
        loss_dict["loss"] = loss
        loss_dict["dip_loss"] = dip_loss
        return DIPVariationalAutoencoderOutput(
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
            dip_loss=dip_loss,
            loss_dict=loss_dict,
        )
