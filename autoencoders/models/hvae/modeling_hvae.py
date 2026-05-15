"""PyTorch implementation of a hierarchical variational autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from ...modeling_outputs import HierarchicalVariationalAutoencoderOutput
from ..base.modeling_vae import BaseVariationalAutoencoderModel
from .configuration_hvae import HierarchicalVariationalAutoencoderConfig


class HierarchicalVariationalAutoencoderModel(BaseVariationalAutoencoderModel):
    """A two-level latent VAE for vector-like embeddings."""

    config_class = HierarchicalVariationalAutoencoderConfig

    def __init__(
        self,
        config: HierarchicalVariationalAutoencoderConfig,
        encoder: str | nn.Module | None = None,
        decoder: str | nn.Module | None = None,
        encoder_config=None,
        decoder_config=None,
    ) -> None:
        super().__init__(config)
        self.encoder, self._encoder_module_type, self._encoder_module_config = self._build_backbone_module(
            module=encoder,
            module_config=encoder_config,
            input_dim=self.config.input_dim,
            output_dim=self.config.latent_dim,
            name="encoder",
        )
        if self.encoder is None:
            self.top_mean_projection = None
            self.top_logvar_projection = None
            self.bottom_mean_projection = None
            self.bottom_logvar_projection = None
        else:
            encoder_output_dim = self._get_backbone_output_dim(self.encoder, "encoder")
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
        self.decoder, self._decoder_module_type, self._decoder_module_config = self._build_backbone_module(
            module=decoder,
            module_config=decoder_config,
            input_dim=self.config.top_latent_dim + self.config.latent_dim,
            output_dim=self.config.input_dim,
            reverse=False,
            name="decoder",
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._require_backbone_module(self.encoder, "encoder")(inputs)
        if (
            self.top_mean_projection is None
            or self.top_logvar_projection is None
            or self.bottom_mean_projection is None
            or self.bottom_logvar_projection is None
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} does not have hierarchical posterior projection layers because no "
                "explicit encoder backbone was provided at initialization time."
            )
        top_mean = self.top_mean_projection(encoded)
        top_logvar = self.top_logvar_projection(encoded)
        top_latents = self.reparameterize(top_mean, top_logvar) if self.training else top_mean
        bottom_inputs = torch.cat([encoded, top_latents], dim=-1)
        bottom_mean = self.bottom_mean_projection(bottom_inputs)
        bottom_logvar = self.bottom_logvar_projection(bottom_inputs)
        return top_mean, top_logvar, bottom_mean, bottom_logvar

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self._require_backbone_module(self.decoder, "decoder")(latents)

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
