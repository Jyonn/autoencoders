"""PyTorch implementation of an information variational autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import InformationVariationalAutoencoderOutput
from ..vae.modeling_vae import VariationalAutoencoderModel
from .configuration_infovae import InformationVariationalAutoencoderConfig


class InformationVariationalAutoencoderModel(VariationalAutoencoderModel):
    """A VAE regularized with an additional MMD prior-matching penalty."""

    config_class = InformationVariationalAutoencoderConfig
    config: InformationVariationalAutoencoderConfig

    def sample_prior(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(batch_size, self.config.latent_dim, device=device, dtype=dtype)

    def compute_mmd_loss(self, latents: torch.Tensor, prior_samples: torch.Tensor) -> torch.Tensor:
        xx = self._compute_kernel(latents, latents)
        yy = self._compute_kernel(prior_samples, prior_samples)
        xy = self._compute_kernel(latents, prior_samples)
        return xx.mean() + yy.mean() - 2.0 * xy.mean()

    def _compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pairwise_distances = torch.cdist(x, y).pow(2)
        kernel = torch.zeros_like(pairwise_distances)
        for bandwidth in self.config.mmd_bandwidths:
            kernel = kernel + torch.exp(-pairwise_distances / (2.0 * bandwidth * bandwidth))
        return kernel / float(len(self.config.mmd_bandwidths))

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> InformationVariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        outputs = super().forward(
            inputs=inputs,
            return_dict=True,
            sample_posterior=sample_posterior,
            global_step=global_step,
            current_epoch=current_epoch,
        )
        prior_samples = self.sample_prior(outputs.latents.shape[0], device=outputs.latents.device, dtype=outputs.latents.dtype)
        mmd_loss = self.compute_mmd_loss(outputs.latents, prior_samples)
        loss = outputs.loss + self.config.mmd_weight * mmd_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, outputs.reconstruction, outputs.latents

        loss_dict = dict(outputs.loss_dict)
        loss_dict["loss"] = loss
        loss_dict["mmd_loss"] = mmd_loss
        return InformationVariationalAutoencoderOutput(
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
            mmd_loss=mmd_loss,
            loss_dict=loss_dict,
        )
