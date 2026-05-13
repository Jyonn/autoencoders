"""PyTorch implementation of a Wasserstein autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import AutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_wae import WassersteinAutoencoderConfig


class WassersteinAutoencoderModel(AutoencoderModel):
    """A deterministic autoencoder regularized with MMD to a Gaussian prior."""

    config_class = WassersteinAutoencoderConfig

    def __init__(self, config: WassersteinAutoencoderConfig) -> None:
        super().__init__(config)

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
    ) -> AutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        latents = self.latent_transform(encoded)
        reconstruction = self.decode(latents)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        prior_samples = self.sample_prior(latents.shape[0], device=latents.device, dtype=latents.dtype)
        mmd_loss = self.compute_mmd_loss(latents, prior_samples)
        loss = reconstruction_loss + self.config.mmd_weight * mmd_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return AutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            reconstruction_loss=reconstruction_loss,
            mmd_loss=mmd_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "mmd_loss": mmd_loss,
            },
        )
