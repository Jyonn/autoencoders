"""PyTorch implementation of a variational autoencoder."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from ...modeling_outputs import AutoencoderOutput
from ..base.modeling_base import BaseAutoencoderModel
from .configuration_vae import VariationalAutoencoderConfig


class VariationalAutoencoderModel(BaseAutoencoderModel):
    """A basic MLP variational autoencoder for vector-like feature inputs."""

    config_class = VariationalAutoencoderConfig

    def __init__(self, config: VariationalAutoencoderConfig) -> None:
        super().__init__(config)
        self.encoder = self._build_encoder()
        encoder_output_dim = self.config.hidden_dims[-1]
        self.mean_projection = nn.Linear(encoder_output_dim, self.config.latent_dim, bias=self.config.use_bias)
        self.logvar_projection = nn.Linear(encoder_output_dim, self.config.latent_dim, bias=self.config.use_bias)
        self.decoder = self._build_decoder()

    def encode(self, features: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        posterior_mean = self.mean_projection(encoded)
        posterior_logvar = self.logvar_projection(encoded)
        return posterior_mean, posterior_logvar

    def decode(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.decoder(latents)

    def reparameterize(self, posterior_mean: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * posterior_logvar)
        epsilon = torch.randn_like(std)
        return posterior_mean + epsilon * std

    def compute_kl_loss(self, posterior_mean: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        kl_per_example = -0.5 * torch.sum(
            1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp(),
            dim=-1,
        )
        return kl_per_example.mean()

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        **kwargs: Any,
    ) -> AutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        model_inputs = self._resolve_inputs(inputs=inputs, features=features)
        posterior_mean, posterior_logvar = self.encode(model_inputs, **kwargs)

        if sample_posterior is None:
            sample_posterior = self.training or not self.config.use_mean_in_eval

        if sample_posterior:
            latents = self.reparameterize(posterior_mean, posterior_logvar)
        else:
            latents = posterior_mean

        reconstruction = self.decode(latents, **kwargs)
        loss_targets = model_inputs if targets is None else targets
        reconstruction_loss = self.compute_loss(reconstruction, loss_targets)
        kl_loss = self.compute_kl_loss(posterior_mean, posterior_logvar)
        loss = reconstruction_loss + self.config.kl_weight * kl_loss
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return AutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=posterior_mean,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            hidden_states={"inputs": model_inputs},
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            },
        )

    def _build_encoder(self) -> nn.Sequential:
        dims = [self.config.input_dim, *self.config.hidden_dims]
        return self._build_mlp(dims)

    def _build_decoder(self) -> nn.Sequential:
        decoder_hidden_dims = self.config.decoder_hidden_dims
        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(self.config.hidden_dims))
        dims = [self.config.latent_dim, *decoder_hidden_dims, self.config.input_dim]
        return self._build_mlp(dims)

    def _build_mlp(self, dims: list[int]) -> nn.Sequential:
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
