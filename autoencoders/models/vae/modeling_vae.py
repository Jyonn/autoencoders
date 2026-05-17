"""PyTorch implementation of a variational autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from ...data.base import TensorSpec
from ..base.modeling_vae import BaseVariationalAutoencoderModel
from .configuration_vae import VariationalAutoencoderConfig


class VariationalAutoencoderModel(BaseVariationalAutoencoderModel):
    """A basic MLP variational autoencoder for vector-like feature inputs."""

    config_class = VariationalAutoencoderConfig
    config: VariationalAutoencoderConfig

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.encoder is None or not isinstance(self.core_spec, TensorSpec):
            self.mean_projection = None
            self.logvar_projection = None
        else:
            encoder_output_dim = self.core_spec.shape[-1]
            if encoder_output_dim is None:
                self.mean_projection = None
                self.logvar_projection = None
            else:
                self.mean_projection = nn.Linear(encoder_output_dim, self.config.latent_dim, bias=True)
                self.logvar_projection = nn.Linear(encoder_output_dim, self.config.latent_dim, bias=True)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encode_backbone(inputs)
        if self.mean_projection is None or self.logvar_projection is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have posterior projection layers because no explicit encoder "
                "backbone was provided at initialization time."
            )
        posterior_mean = self.mean_projection(encoded)
        posterior_logvar = self.logvar_projection(encoded)
        return posterior_mean, posterior_logvar
