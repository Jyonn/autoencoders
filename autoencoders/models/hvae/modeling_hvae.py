"""PyTorch implementation of a hierarchical variational autoencoder."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ...modeling_outputs import HierarchicalVariationalAutoencoderOutput
from ..base.modeling_vae import BaseVariationalAutoencoderModel
from .configuration_hvae import HierarchicalVariationalAutoencoderConfig


class HierarchicalVariationalAutoencoderModel(BaseVariationalAutoencoderModel):
    """A two-level latent VAE for vector-like embeddings."""

    config_class = HierarchicalVariationalAutoencoderConfig

    def __init__(self, config: HierarchicalVariationalAutoencoderConfig) -> None:
        super().__init__(config)
        self.encoder = self._build_encoder()
        encoder_output_dim = self.config.hidden_dims[-1]
        self.top_mean_projection = nn.Linear(encoder_output_dim, self.config.top_latent_dim, bias=self.config.use_bias)
        self.top_logvar_projection = nn.Linear(encoder_output_dim, self.config.top_latent_dim, bias=self.config.use_bias)
        self.bottom_mean_projection = nn.Linear(
            encoder_output_dim + self.config.top_latent_dim,
            self.config.latent_dim,
            bias=self.config.use_bias,
        )
        self.bottom_logvar_projection = nn.Linear(
            encoder_output_dim + self.config.top_latent_dim,
            self.config.latent_dim,
            bias=self.config.use_bias,
        )
        self.decoder = self._build_decoder()

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(inputs)
        top_mean = self.top_mean_projection(encoded)
        top_logvar = self.top_logvar_projection(encoded)
        top_latents = self.reparameterize(top_mean, top_logvar) if self.training else top_mean
        bottom_inputs = torch.cat([encoded, top_latents], dim=-1)
        bottom_mean = self.bottom_mean_projection(bottom_inputs)
        bottom_logvar = self.bottom_logvar_projection(bottom_inputs)
        return top_mean, top_logvar, bottom_mean, bottom_logvar

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> HierarchicalVariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        top_mean, top_logvar, bottom_mean, bottom_logvar = self.encode(inputs)
        if sample_posterior is None:
            sample_posterior = self.training or not self.config.use_mean_in_eval

        if sample_posterior:
            top_latents = self.reparameterize(top_mean, top_logvar)
            bottom_latents = self.reparameterize(bottom_mean, bottom_logvar)
        else:
            top_latents = top_mean
            bottom_latents = bottom_mean

        latents = torch.cat([top_latents, bottom_latents], dim=-1)
        reconstruction = self.decode(latents)
        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        top_kl_loss = self.compute_kl_loss(top_mean, top_logvar)
        bottom_kl_loss = self.compute_kl_loss(bottom_mean, bottom_logvar)
        kl_loss = top_kl_loss + bottom_kl_loss
        free_bits_kl_loss = self.compute_free_bits_kl_loss(
            torch.cat([top_mean, bottom_mean], dim=-1),
            torch.cat([top_logvar, bottom_logvar], dim=-1),
        )
        effective_kl_weight = self.get_current_kl_weight(global_step=global_step, current_epoch=current_epoch)
        loss = self.compute_total_loss(reconstruction_loss, free_bits_kl_loss, kl_weight=effective_kl_weight)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return HierarchicalVariationalAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=bottom_mean,
            posterior_mean=torch.cat([top_mean, bottom_mean], dim=-1),
            posterior_logvar=torch.cat([top_logvar, bottom_logvar], dim=-1),
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            free_bits_kl_loss=free_bits_kl_loss,
            effective_kl_weight=effective_kl_weight,
            hierarchical_kl_loss=kl_loss,
            hidden_states={
                "top_latents": top_latents,
                "bottom_latents": bottom_latents,
                "top_mean": top_mean,
                "top_logvar": top_logvar,
                "bottom_mean": bottom_mean,
                "bottom_logvar": bottom_logvar,
            },
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "free_bits_kl_loss": free_bits_kl_loss,
                "effective_kl_weight": loss.new_tensor(effective_kl_weight),
                "hierarchical_kl_loss": kl_loss,
            },
        )

    def _build_encoder(self) -> nn.Sequential:
        dims = [self.config.input_dim, *self.config.hidden_dims]
        return self._build_mlp(dims)

    def _build_decoder(self) -> nn.Sequential:
        decoder_hidden_dims = self.config.decoder_hidden_dims
        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(self.config.hidden_dims))
        dims = [self.config.top_latent_dim + self.config.latent_dim, *decoder_hidden_dims, self.config.input_dim]
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
