"""Base PyTorch model for autoencoder-family implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import AutoencoderExport, BaseAutoencoderOutput
from ...modeling_utils import PreTrainedAutoencoderModel
from ...modules import BaseAutoencoderModule, get_module_class
from .configuration_base import BaseAutoencoderConfig


class BaseAutoencoderModel(PreTrainedAutoencoderModel, ABC):
    """Shared model skeleton for deterministic autoencoders."""

    config_class = BaseAutoencoderConfig
    requires_grad_in_eval = False
    min_input_rank = 2

    def __init__(self, config: BaseAutoencoderConfig) -> None:
        super().__init__(config)
        self._encoder_module_type: str | None = None
        self._decoder_module_type: str | None = None
        self._encoder_module_config = None
        self._decoder_module_config = None

    def get_serializable_module_specs(self) -> dict[str, dict[str, object]]:
        module_specs: dict[str, dict[str, object]] = {}
        for name in ("encoder", "decoder"):
            module_type = getattr(self, f"_{name}_module_type")
            module_config = getattr(self, f"_{name}_module_config")
            if module_type is None:
                continue
            if module_type == "external":
                module_specs[name] = {"module_type": "external"}
                continue
            module_specs[name] = {
                "module_type": module_type,
                "module_config": module_config.to_dict(),
            }
        return module_specs

    def _coerce_module_config(self, module_class, module_config):
        config_class = module_class.config_class
        if isinstance(module_config, config_class):
            return module_config
        if isinstance(module_config, dict):
            return config_class(**module_config)
        return config_class.from_dict(module_config.to_dict())

    def _build_backbone_module(
        self,
        *,
        module: str | nn.Module | None,
        module_config,
        input_dim: int,
        output_dim: int,
        reverse: bool = False,
        name: str,
    ):
        if module is None:
            warnings.warn(
                f"{self.__class__.__name__} was initialized without an explicit {name} backbone. "
                "Pass a built-in module name "
                "such as `mlp` together with its module config, or inject a custom nn.Module.",
                stacklevel=3,
            )
            return None, None, None
        if isinstance(module, nn.Module):
            return module, "external", None
        if module_config is None:
            raise ValueError(
                f"{self.__class__.__name__} received built-in {name}={module!r} without `{name}_config`. "
                f"Please provide `{name}_config` as a plain dict."
            )

        module_name = module
        module_class = get_module_class(module_name)
        resolved_config = self._coerce_module_config(module_class, module_config)
        built_module = module_class(
            resolved_config,
            input_dim=input_dim,
            latent_dim=output_dim,
            reverse=reverse,
        )
        return built_module, module_name, resolved_config

    def _require_backbone_module(self, module: nn.Module | None, name: str) -> nn.Module:
        if module is None:
            article = "an" if name[0].lower() in {"a", "e", "i", "o", "u"} else "a"
            raise RuntimeError(
                f"{self.__class__.__name__} does not have {article} {name} backbone. "
                f"Construct the model with `{name}=...` and the corresponding `{name}_config=...`."
            )
        return module

    def _build_decoder_backbone_module(
        self,
        *,
        encoder_module: nn.Module | None,
        encoder_module_type: str | None,
        encoder_module_config,
        module: str | nn.Module | None,
        module_config,
        input_dim: int,
        output_dim: int,
        name: str = "decoder",
    ):
        if module is not None:
            return self._build_backbone_module(
                module=module,
                module_config=module_config,
                input_dim=input_dim,
                output_dim=output_dim,
                reverse=False,
                name=name,
            )

        if encoder_module is None:
            warnings.warn(
                f"{self.__class__.__name__} was initialized without an explicit {name} backbone because no "
                "encoder backbone was available to derive it from.",
                stacklevel=3,
            )
            return None, None, None

        if not isinstance(encoder_module, BaseAutoencoderModule):
            raise ValueError(
                f"{self.__class__.__name__} cannot infer {name} from encoder={encoder_module.__class__.__name__}. "
                f"When `{name}=None`, the encoder must be a built-in or custom `{BaseAutoencoderModule.__name__}`."
            )

        derived_module = encoder_module.__class__(
            encoder_module.config,
            input_dim=input_dim,
            latent_dim=output_dim,
            reverse=True,
        )
        if encoder_module_type in {None, "external"}:
            return derived_module, None, None
        return derived_module, encoder_module_type, encoder_module_config

    def _get_backbone_output_dim(self, module: nn.Module | None, name: str) -> int:
        resolved_module = self._require_backbone_module(module, name)
        if hasattr(resolved_module, "get_output_dim"):
            return int(resolved_module.get_output_dim())  # type: ignore[call-arg]
        if hasattr(resolved_module, "out_features"):
            return int(resolved_module.out_features)
        raise RuntimeError(
            f"{self.__class__.__name__} requires {name} to expose `get_output_dim()` "
            "or an `out_features` attribute."
        )

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into latent representations."""

    def latent_transform(self, encoded: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses such as VAE or VQ-VAE."""
        return encoded

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back into feature space."""

    def reconstruct(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(inputs=inputs, return_dict=True)
        return outputs.reconstruction

    def validate_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim < self.min_input_rank:
            raise ValueError(
                f"{self.__class__.__name__} expects inputs with rank >= {self.min_input_rank}, "
                f"but received shape {tuple(inputs.shape)}."
            )

    def get_epoch_metrics(
        self,
        *,
        global_step: int | None = None,
        current_epoch: int | None = None,
    ) -> dict[str, float | int]:
        del global_step, current_epoch
        return {}

    def consume_dead_code_reset_count(self) -> int:
        return 0

    def export(
        self,
        inputs: torch.Tensor,
        include_reconstruction: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> AutoencoderExport:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs=inputs, return_dict=True)
        if was_training:
            self.train()
        return self._build_export(
            inputs=inputs,
            outputs=outputs,
            include_reconstruction=include_reconstruction,
            metadata=metadata,
        )

    def compute_loss(self, reconstruction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.reconstruction_loss == "mse":
            return F.mse_loss(reconstruction, targets)
        if self.config.reconstruction_loss == "l1":
            return F.l1_loss(reconstruction, targets)
        raise ValueError(f"Unsupported reconstruction loss: {self.config.reconstruction_loss}")

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> BaseAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        self.validate_inputs(inputs)
        encoded = self.encode(inputs)
        latents = self.latent_transform(encoded)
        reconstruction = self.decode(latents)

        loss = self.compute_loss(reconstruction, inputs)
        use_return_dict = self.config.return_dict if return_dict is None else return_dict

        if not use_return_dict:
            return loss, reconstruction, latents

        return BaseAutoencoderOutput(
            loss=loss,
            reconstruction=reconstruction,
            latents=latents,
            encoded=encoded,
            loss_dict={"reconstruction_loss": loss},
        )

    def _build_export(
        self,
        *,
        inputs: torch.Tensor,
        outputs: BaseAutoencoderOutput,
        include_reconstruction: bool,
        metadata: dict[str, object] | None,
    ) -> AutoencoderExport:
        export_metadata: dict[str, object] = {
            "input_shape": list(inputs.shape),
            "latent_shape": list(outputs.latents.shape) if outputs.latents is not None else None,
        }
        if metadata is not None:
            export_metadata.update(metadata)

        return AutoencoderExport(
            model_type=self.config.model_type,
            latents=outputs.latents,
            reconstruction=outputs.reconstruction if include_reconstruction else None,
            encoded=outputs.encoded,
            metadata=export_metadata,
        )
