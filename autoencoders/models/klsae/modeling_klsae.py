"""PyTorch implementation of a KL sparse autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import KLSparseAutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_klsae import KLSparseAutoencoderConfig


class KLSparseAutoencoderModel(AutoencoderModel):
    """An autoencoder with KL divergence sparsity regularization on latent activity."""

    config_class = KLSparseAutoencoderConfig
    config: KLSparseAutoencoderConfig

    def compute_kl_sparsity_loss(self, latents: torch.Tensor) -> torch.Tensor:
        rho = torch.full(
            (latents.shape[-1],),
            fill_value=self.config.target_activation,
            device=latents.device,
            dtype=latents.dtype,
        )
        rho_hat = torch.sigmoid(latents).mean(dim=0).clamp(min=1e-6, max=1 - 1e-6)
        return (
            rho * torch.log(rho / rho_hat)
            + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        ).sum()

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> KLSparseAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        core_inputs = self.project_to_core(encoded)
        latents = core_inputs
        reconstruction = self.decode(self.project_from_core(latents))

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        kl_sparsity_loss = self.compute_kl_sparsity_loss(latents)
        loss = reconstruction_loss + self.config.sparsity_weight * kl_sparsity_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return KLSparseAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            reconstruction_loss=reconstruction_loss,
            kl_sparsity_loss=kl_sparsity_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_sparsity_loss": kl_sparsity_loss,
            },
        )
