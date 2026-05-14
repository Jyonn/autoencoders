"""PyTorch implementation of an MMD-VAE."""

from __future__ import annotations

from ...modeling_outputs import MMDVariationalAutoencoderOutput
from ..infovae.modeling_infovae import InformationVariationalAutoencoderModel
from .configuration_mmdvae import MMDVariationalAutoencoderConfig


class MMDVariationalAutoencoderModel(InformationVariationalAutoencoderModel):
    """A VAE regularized primarily through MMD prior matching."""

    config_class = MMDVariationalAutoencoderConfig

    def forward(
        self,
        inputs,
        return_dict: bool | None = None,
        sample_posterior: bool | None = None,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ):
        outputs = super().forward(
            inputs=inputs,
            return_dict=True,
            sample_posterior=sample_posterior,
            global_step=global_step,
            current_epoch=current_epoch,
        )
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        if not use_return_dict:
            return outputs.loss, outputs.reconstruction, outputs.latents
        return MMDVariationalAutoencoderOutput(
            loss=outputs.loss,
            reconstruction=outputs.reconstruction,
            latents=outputs.latents,
            encoded=outputs.encoded,
            posterior_mean=outputs.posterior_mean,
            posterior_logvar=outputs.posterior_logvar,
            reconstruction_loss=outputs.reconstruction_loss,
            kl_loss=outputs.kl_loss,
            free_bits_kl_loss=outputs.free_bits_kl_loss,
            effective_kl_weight=outputs.effective_kl_weight,
            mmd_loss=outputs.mmd_loss,
            loss_dict=dict(outputs.loss_dict),
        )
