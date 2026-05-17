"""Base PyTorch model for autoencoder-family implementations."""

from __future__ import annotations

from abc import ABC
import warnings
import torch
import torch.nn.functional as F
from torch import nn

from ...data.base import DataSpec, TensorSpec
from ...modeling_outputs import AutoencoderExport, BaseAutoencoderOutput
from ...modeling_utils import PreTrainedAutoencoderModel
from ...modules import BaseAutoencoderModule, get_module_class
from .configuration_base import BaseAutoencoderConfig


class BaseAutoencoderModel(PreTrainedAutoencoderModel, ABC):
    """Shared model skeleton for deterministic autoencoders."""

    config_class = BaseAutoencoderConfig
    config: BaseAutoencoderConfig
    requires_grad_in_eval = False
    min_input_rank = 2

    def __init__(self, **kwargs: object) -> None:
        config = kwargs.pop("config")
        self.sample_spec = kwargs.pop("sample_spec", None)
        super().__init__(config=config)
        self.encoder: nn.Module | None = None
        self.decoder: nn.Module | None = None
        self.encoder_input_spec: DataSpec | None = None
        self.encoder_output_spec: DataSpec | None = None
        self.core_output_spec: DataSpec | None = None
        self.decoder_input_spec: DataSpec | None = None
        self.decoder_output_spec: DataSpec | None = None
        self.encoder_to_core_projection: nn.Module | None = None
        self.core_to_decoder_projection: nn.Module | None = None
        self.decoder_output_projection: nn.Module | None = None
        self._encoder_module_type: str | None = None
        self._decoder_module_type: str | None = None
        self._encoder_module_config = None
        self._decoder_module_config = None
        self._decoder_is_auto = False
        self._initialize_backbones(**kwargs)

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

    def _build_backbone_module(
        self,
        *,
        module: str | nn.Module | None,
        module_config,
        input_spec: DataSpec,
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
        resolved_config = module_class.config_class(**module_config)
        built_module = module_class(
            config=resolved_config,
            input_spec=input_spec,
            reverse=reverse,
        )
        return built_module, module_name, resolved_config

    def get_encoder_input_dim(self) -> int:
        return int(self.config.input_dim)

    def get_core_latent_dim(self) -> int | None:
        return None if self.config.latent_dim is None else int(self.config.latent_dim)

    def get_decoder_output_dim(self) -> int:
        return int(self.config.input_dim)

    def get_encoder_input_spec(self) -> DataSpec:
        if self.sample_spec is not None:
            return self.sample_spec
        return TensorSpec(shape=(self.get_encoder_input_dim(),))

    def get_decoder_input_spec(self) -> DataSpec:
        if self.encoder_output_spec is not None:
            return self.encoder_output_spec
        core_latent_dim = self.get_core_latent_dim()
        if core_latent_dim is not None:
            return TensorSpec(shape=(core_latent_dim,))
        return TensorSpec(shape=(self.get_encoder_input_dim(),))

    def get_reconstruction_target_spec(self) -> DataSpec:
        if self.sample_spec is not None:
            return self.sample_spec
        return TensorSpec(shape=(self.get_decoder_output_dim(),))

    def get_latent_output_spec(self) -> DataSpec | None:
        return self.core_output_spec

    def _resolve_tensor_feature_dim(self, spec: DataSpec, *, name: str) -> int:
        if not isinstance(spec, TensorSpec):
            raise RuntimeError(f"{self.__class__.__name__} requires {name} to be a TensorSpec.")
        if not spec.shape or spec.shape[-1] is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires {name} to expose a concrete final feature dimension."
            )
        return int(spec.shape[-1])

    def _build_projected_spec(self, spec: DataSpec, *, feature_dim: int, name: str) -> DataSpec:
        if not isinstance(spec, TensorSpec):
            raise RuntimeError(f"{self.__class__.__name__} requires {name} to be a TensorSpec.")
        return TensorSpec(shape=(*spec.shape[:-1], feature_dim))

    def _build_linear_projection(
        self,
        input_spec: DataSpec | None,
        output_spec: DataSpec | None,
        *,
        name: str,
    ) -> nn.Linear | None:
        if input_spec is None or output_spec is None:
            return None
        input_dim = self._resolve_tensor_feature_dim(input_spec, name=f"{name} input spec")
        output_dim = self._resolve_tensor_feature_dim(output_spec, name=f"{name} output spec")
        if input_dim == output_dim:
            return None
        return nn.Linear(input_dim, output_dim, bias=True)

    def _initialize_backbones(self, **kwargs: object) -> None:
        encoder = kwargs.pop("encoder", None)
        decoder = kwargs.pop("decoder", None)
        encoder_config = kwargs.pop("encoder_config", None)
        decoder_config = kwargs.pop("decoder_config", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"{self.__class__.__name__} received unexpected keyword arguments: {unknown}")

        self.encoder_input_spec = self.get_encoder_input_spec()
        self.encoder, self._encoder_module_type, self._encoder_module_config = self._build_backbone_module(
            module=encoder,
            module_config=encoder_config,
            input_spec=self.encoder_input_spec,
            name="encoder",
        )
        if isinstance(self.encoder, BaseAutoencoderModule):
            self.encoder_output_spec = self.encoder.output_spec
        else:
            self.encoder_output_spec = None
        core_latent_dim = self.get_core_latent_dim()
        if core_latent_dim is not None and self.encoder_output_spec is not None:
            self.core_output_spec = self._build_projected_spec(
                self.encoder_output_spec,
                feature_dim=core_latent_dim,
                name="encoder output spec",
            )
        else:
            self.core_output_spec = self.encoder_output_spec
        self.encoder_to_core_projection = self._build_linear_projection(
            self.encoder_output_spec,
            self.core_output_spec,
            name="encoder-to-core projection",
        )
        self.decoder_input_spec = self.get_decoder_input_spec()
        (
            self.decoder,
            self._decoder_module_type,
            self._decoder_module_config,
            self._decoder_is_auto,
        ) = self._build_decoder_backbone_module(
            encoder_module=self.encoder,
            encoder_module_type=self._encoder_module_type,
            encoder_module_config=self._encoder_module_config,
            module=decoder,
            module_config=decoder_config,
            input_spec=self.decoder_input_spec,
            name="decoder",
        )
        if isinstance(self.decoder, BaseAutoencoderModule):
            self.decoder_output_spec = self.decoder.output_spec
        else:
            self.decoder_output_spec = None
        self.core_to_decoder_projection = self._build_linear_projection(
            self.get_latent_output_spec(),
            self.decoder_input_spec,
            name="core-to-decoder projection",
        )
        self.decoder_output_projection = self._build_linear_projection(
            self.decoder_output_spec,
            self.get_reconstruction_target_spec(),
            name="decoder-output projection",
        )

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
        input_spec: DataSpec,
        name: str = "decoder",
    ):
        if module is not None:
            built_module, module_type, resolved_config = self._build_backbone_module(
                module=module,
                module_config=module_config,
                input_spec=input_spec,
                reverse=False,
                name=name,
            )
            return built_module, module_type, resolved_config, False

        if encoder_module is None:
            warnings.warn(
                f"{self.__class__.__name__} was initialized without an explicit {name} backbone because no "
                "encoder backbone was available to derive it from.",
                stacklevel=3,
            )
            return None, None, None, False

        if not isinstance(encoder_module, BaseAutoencoderModule):
            raise ValueError(
                f"{self.__class__.__name__} cannot infer {name} from encoder={encoder_module.__class__.__name__}. "
                f"When `{name}=None`, the encoder must be a built-in or custom `{BaseAutoencoderModule.__name__}`."
            )

        derived_module = encoder_module.__class__(
            config=encoder_module.config,
            input_spec=input_spec,
            reverse=True,
        )
        if encoder_module_type in {None, "external"}:
            return derived_module, None, None, True
        return derived_module, encoder_module_type, encoder_module_config, True

    def _encode_backbone(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._require_backbone_module(self.encoder, "encoder")(inputs)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into latent representations."""
        encoded = self._encode_backbone(inputs)
        if self.encoder_to_core_projection is not None:
            encoded = self.encoder_to_core_projection(encoded)
        return encoded

    def latent_transform(self, encoded: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses such as VAE or VQ-VAE."""
        return encoded

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back into feature space."""
        if self.core_to_decoder_projection is not None:
            latents = self.core_to_decoder_projection(latents)
        decoded = self._require_backbone_module(self.decoder, "decoder")(latents)
        if self.decoder_output_projection is not None:
            decoded = self.decoder_output_projection(decoded)
        return decoded

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
