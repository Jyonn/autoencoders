"""PyTorch implementation of a sparse autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import SparseAutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_sae import SparseAutoencoderConfig


class SparseAutoencoderModel(AutoencoderModel):
    """A deterministic autoencoder with latent L1 sparsity regularization."""

    config_class = SparseAutoencoderConfig

    def __init__(
        self,
        config: SparseAutoencoderConfig,
        encoder=None,
        decoder=None,
        encoder_config=None,
        decoder_config=None,
    ) -> None:
        super().__init__(
            config,
            encoder=encoder,
            decoder=decoder,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

    def compute_sparsity_loss(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.abs().mean()

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> SparseAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        latents = self.latent_transform(encoded)
        reconstruction = self.decode(latents)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        sparsity_loss = self.compute_sparsity_loss(latents)
        loss = reconstruction_loss + self.config.sparsity_weight * sparsity_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return SparseAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "sparsity_loss": sparsity_loss,
            },
        )
