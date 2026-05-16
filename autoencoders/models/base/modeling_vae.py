"""Base model shared by variational autoencoder families."""

from __future__ import annotations

from abc import abstractmethod

import torch

from ...modeling_outputs import AutoencoderExport, VariationalAutoencoderOutput
from .configuration_vae import BaseVariationalAutoencoderConfig
from .modeling_base import BaseAutoencoderModel


class BaseVariationalAutoencoderModel(BaseAutoencoderModel):
    """Shared VAE forward path and KL utilities."""

    config_class = BaseVariationalAutoencoderConfig
    config: BaseVariationalAutoencoderConfig

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs into posterior parameters."""

    def reparameterize(self, posterior_mean: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * posterior_logvar)
        epsilon = torch.randn_like(std)
        return posterior_mean + epsilon * std

    def sample_latents(
        self,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        sample_posterior: bool | None = None,
    ) -> torch.Tensor:
        if sample_posterior is None:
            sample_posterior = self.training or not self.config.use_mean_in_eval
        if sample_posterior:
            return self.reparameterize(posterior_mean, posterior_logvar)
        return posterior_mean

    def compute_kl_loss(self, posterior_mean: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        kl_per_example = -0.5 * torch.sum(
            1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp(),
            dim=-1,
        )
        return kl_per_example.mean()

    def compute_free_bits_kl_loss(self, posterior_mean: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        kl_per_dim = -0.5 * (
            1
            + posterior_logvar
            - posterior_mean.pow(2)
            - posterior_logvar.exp()
        )
        mean_kl_per_dim = kl_per_dim.mean(dim=0)
        return torch.clamp(mean_kl_per_dim, min=float(self.config.free_bits)).sum()

    def get_current_kl_weight(
        self,
        *,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> float:
        del global_step
        target_weight = float(self.config.kl_weight)
        warmup_epochs = int(self.config.kl_warmup_epochs)
        start_weight = float(self.config.kl_start_weight)

        if warmup_epochs <= 0:
            return target_weight
        if current_epoch is None:
            if not self.training:
                return target_weight
            raise ValueError("current_epoch must be provided when kl_warmup_epochs is enabled during training.")
        if warmup_epochs == 1:
            return target_weight
        if current_epoch <= 1:
            return start_weight

        progress = min((current_epoch - 1) / (warmup_epochs - 1), 1.0)
        return start_weight + progress * (target_weight - start_weight)

    def get_epoch_metrics(
        self,
        *,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> dict[str, float]:
        return {
            "kl_weight": self.get_current_kl_weight(
                global_step=global_step,
                current_epoch=current_epoch,
            ),
        }

    def compute_total_loss(
        self,
        reconstruction_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        *,
        kl_weight: float | None = None,
    ) -> torch.Tensor:
        effective_kl_weight = self.config.kl_weight if kl_weight is None else kl_weight
        return reconstruction_loss + effective_kl_weight * kl_loss

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> VariationalAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        self.validate_inputs(inputs)
        posterior_mean, posterior_logvar = self.encode(inputs)
        latents = self.sample_latents(
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            sample_posterior=sample_posterior,
        )
        reconstruction = self.decode(latents)
        reconstruction_loss = self.compute_loss(reconstruction, inputs)
        kl_loss = self.compute_kl_loss(posterior_mean, posterior_logvar)
        free_bits_kl_loss = self.compute_free_bits_kl_loss(posterior_mean, posterior_logvar)
        effective_kl_weight = self.get_current_kl_weight(global_step=global_step, current_epoch=current_epoch)
        loss = self.compute_total_loss(reconstruction_loss, free_bits_kl_loss, kl_weight=effective_kl_weight)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return VariationalAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=posterior_mean,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            free_bits_kl_loss=free_bits_kl_loss,
            effective_kl_weight=effective_kl_weight,
            loss_dict={
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "free_bits_kl_loss": free_bits_kl_loss,
                "effective_kl_weight": loss.new_tensor(effective_kl_weight),
            },
        )

    def _build_export(
        self,
        *,
        inputs: torch.Tensor,
        outputs: VariationalAutoencoderOutput,
        include_reconstruction: bool,
        metadata: dict[str, object] | None,
    ) -> AutoencoderExport:
        artifact = super()._build_export(
            inputs=inputs,
            outputs=outputs,
            include_reconstruction=include_reconstruction,
            metadata=metadata,
        )
        artifact.posterior_mean = outputs.posterior_mean
        artifact.posterior_logvar = outputs.posterior_logvar
        return artifact
