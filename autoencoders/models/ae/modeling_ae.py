"""PyTorch implementation of a basic feed-forward autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from ..base.modeling_base import BaseAutoencoderModel
from .configuration_ae import AutoencoderConfig


class AutoencoderModel(BaseAutoencoderModel):
    """A deterministic MLP autoencoder for vector-like feature inputs."""

    config_class = AutoencoderConfig

    def __init__(
        self,
        config: AutoencoderConfig,
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
            output_dim=self.config.latent_dim,
        )
        self.decoder, self._decoder_module_type, self._decoder_module_config = self._build_backbone_module(
            module=decoder,
            module_config=decoder_config,
            default_module_name="mlp",
            default_module_config=self._create_default_decoder_module_config(),
            input_dim=self.config.latent_dim,
            output_dim=self.config.input_dim,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
