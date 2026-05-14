"""PyTorch implementation of a contractive autoencoder."""

from __future__ import annotations

import torch

from ...modeling_outputs import ContractiveAutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_cae import ContractiveAutoencoderConfig


class ContractiveAutoencoderModel(AutoencoderModel):
    """A deterministic autoencoder with a contractive encoder penalty."""

    config_class = ContractiveAutoencoderConfig
    requires_grad_in_eval = True

    def __init__(
        self,
        config: ContractiveAutoencoderConfig,
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

    def compute_contractive_loss(self, encoded: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        contractive_penalty = torch.zeros(encoded.shape[0], device=encoded.device, dtype=encoded.dtype)

        for latent_index in range(encoded.shape[-1]):
            gradients = torch.autograd.grad(
                encoded[:, latent_index].sum(),
                inputs,
                retain_graph=True,
                create_graph=self.training,
            )[0]
            contractive_penalty = contractive_penalty + gradients.pow(2).sum(dim=-1)

        return contractive_penalty.mean()

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> ContractiveAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            encoder_inputs = inputs.detach().clone().requires_grad_(True)
        else:
            encoder_inputs = inputs

        encoded = self.encode(encoder_inputs)
        latents = self.latent_transform(encoded)
        reconstruction = self.decode(latents)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        if torch.is_grad_enabled():
            contractive_loss = self.compute_contractive_loss(encoded, encoder_inputs)
        else:
            contractive_loss = torch.zeros_like(reconstruction_loss)
        loss = reconstruction_loss + self.config.contractive_weight * contractive_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return ContractiveAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            reconstruction_loss=reconstruction_loss,
            contractive_loss=contractive_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "contractive_loss": contractive_loss,
            },
        )
