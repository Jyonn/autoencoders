"""PyTorch implementation of a basic feed-forward autoencoder."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..base.modeling_base import BaseAutoencoderModel
from .configuration_ae import AutoencoderConfig


class AutoencoderModel(BaseAutoencoderModel):
    """A deterministic MLP autoencoder for vector-like feature inputs."""

    config_class = AutoencoderConfig

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__(config)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def _build_encoder(self) -> nn.Sequential:
        dims = [self.config.input_dim, *self.config.hidden_dims, self.config.latent_dim]
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
