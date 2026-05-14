"""PyTorch implementation of an adversarial autoencoder."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import AdversarialAutoencoderOutput
from ..ae.modeling_ae import AutoencoderModel
from .configuration_aae import AdversarialAutoencoderConfig


class AdversarialAutoencoderModel(AutoencoderModel):
    """A deterministic autoencoder regularized by latent adversarial matching."""

    config_class = AdversarialAutoencoderConfig

    def __init__(self, config: AdversarialAutoencoderConfig) -> None:
        super().__init__(config)
        self.discriminator = self._build_discriminator()

    def sample_prior(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(batch_size, self.config.latent_dim, device=device, dtype=dtype)

    def discriminate(self, latents: torch.Tensor) -> torch.Tensor:
        return self.discriminator(latents)

    def compute_adversarial_loss(self, latents: torch.Tensor) -> torch.Tensor:
        logits = self.discriminate(latents)
        targets = torch.ones_like(logits)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def compute_discriminator_loss(self, latents: torch.Tensor, prior_samples: torch.Tensor) -> torch.Tensor:
        fake_logits = self.discriminate(latents)
        real_logits = self.discriminate(prior_samples)
        fake_targets = torch.zeros_like(fake_logits)
        real_targets = torch.ones_like(real_logits)
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
        return 0.5 * (fake_loss + real_loss)

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> AdversarialAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        latents = self.latent_transform(encoded)
        reconstruction = self.decode(latents)

        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        prior_samples = self.sample_prior(latents.shape[0], device=latents.device, dtype=latents.dtype)
        adversarial_loss = self.compute_adversarial_loss(latents)
        discriminator_loss = self.compute_discriminator_loss(latents.detach(), prior_samples)
        loss = reconstruction_loss + self.config.adversarial_weight * adversarial_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return AdversarialAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            reconstruction_loss=reconstruction_loss,
            adversarial_loss=adversarial_loss,
            discriminator_loss=discriminator_loss,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "adversarial_loss": adversarial_loss,
                "discriminator_loss": discriminator_loss,
            },
        )

    def _build_discriminator(self) -> nn.Sequential:
        dims = [self.config.latent_dim, *self.config.discriminator_hidden_dims, 1]
        layers: list[nn.Module] = []
        activation_factory = self._get_activation_factory()

        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim, bias=self.config.use_bias))
            is_last_layer = index == len(dims) - 2
            if not is_last_layer:
                layers.append(activation_factory())

        return nn.Sequential(*layers)

    def _get_activation_factory(self) -> Callable[[], nn.Module]:
        activations: dict[str, Callable[[], nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        return activations[self.config.activation]
