"""PyTorch implementation of a variational autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from ..base.modeling_vae import BaseVariationalAutoencoderModel
from .configuration_vae import VariationalAutoencoderConfig


class VariationalAutoencoderModel(BaseVariationalAutoencoderModel):
    """A basic MLP variational autoencoder for vector-like feature inputs."""

    config_class = VariationalAutoencoderConfig

    def __init__(
        self,
        config: VariationalAutoencoderConfig,
        encoder: str | nn.Module | None = None,
        decoder: str | nn.Module | None = None,
        encoder_config=None,
        decoder_config=None,
    ) -> None:
        super().__init__(config)
        self.encoder, self._encoder_module_type, self._encoder_module_config = self._build_backbone_module(
            module=encoder,
            module_config=encoder_config,
            default_module_name="mlp",
            default_module_config=self._create_default_encoder_module_config(),
            input_dim=self.config.input_dim,
            output_dim=self.config.hidden_dims[-1],
        )
        encoder_output_dim = self.config.hidden_dims[-1]
        self.mean_projection = nn.Linear(encoder_output_dim, self.config.latent_dim, bias=self.config.use_bias)
        self.logvar_projection = nn.Linear(encoder_output_dim, self.config.latent_dim, bias=self.config.use_bias)
        self.decoder, self._decoder_module_type, self._decoder_module_config = self._build_backbone_module(
            module=decoder,
            module_config=decoder_config,
            default_module_name="mlp",
            default_module_config=self._create_default_decoder_module_config(),
            input_dim=self.config.latent_dim,
            output_dim=self.config.input_dim,
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(inputs)
        posterior_mean = self.mean_projection(encoded)
        posterior_logvar = self.logvar_projection(encoded)
        return posterior_mean, posterior_logvar

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
