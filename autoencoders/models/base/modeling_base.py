"""Base PyTorch model for autoencoder-family implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import warnings
import torch
import torch.nn.functional as F
from torch import nn

from ...data.base import DataSpec, TensorSpec
from ...modeling_outputs import AutoencoderExport, BaseAutoencoderOutput
from ...modeling_utils import PreTrainedAutoencoderModel
from ...modules import BaseAutoencoderModule, ModuleTraceStep, get_module_class
from .configuration_base import BaseAutoencoderConfig


@dataclass(frozen=True)
class PipelineTraceStep:
    """A single stage in the sample -> encoder -> core -> decoder pipeline."""

    name: str
    input_spec: DataSpec
    output_spec: DataSpec
    children: list[ModuleTraceStep] | None = None


class BaseAutoencoderModel(PreTrainedAutoencoderModel, ABC):
    """Shared model skeleton for deterministic autoencoders."""

    config_class = BaseAutoencoderConfig
    config: BaseAutoencoderConfig
    requires_grad_in_eval = False

    encoder: BaseAutoencoderModule | None
    decoder: BaseAutoencoderModule | None

    def __init__(
            self,
            config: BaseAutoencoderConfig,
            sample_spec: DataSpec | None = None,
            encoder: str | BaseAutoencoderModule | None = None,
            decoder: str | BaseAutoencoderModule | None = None,
            encoder_config: dict | None = None,
            decoder_config: dict | None = None,
            **kwargs: object
    ) -> None:
        super().__init__(config=config)
        self.config = config
        if sample_spec is not None:
            self.sample_spec = sample_spec
        elif self.config.input_dim is not None:
            self.sample_spec = TensorSpec(shape=(int(self.config.input_dim),))
        else:
            raise ValueError(
                f"{self.__class__.__name__} requires `sample_spec` when `config.input_dim` is not provided."
            )
        self._encoder_module_type = None
        self._encoder_module_config = None
        self._decoder_module_type = None
        self._decoder_module_config = None

        self.encoder = None
        self.encoder_config = None
        if isinstance(encoder, str):
            encoder_class = get_module_class(encoder)
            encoder_config_class = encoder_class.config_class
            if encoder_config is None:
                raise ValueError(
                    f"{self.__class__.__name__} received built-in encoder={encoder!r} without `encoder_config`."
                )
            self.encoder_config = (
                encoder_config
                if isinstance(encoder_config, encoder_config_class)
                else encoder_config_class(**encoder_config)
            )
            self.encoder = encoder_class(config=self.encoder_config, input_spec=self.sample_spec)
            self._encoder_module_type = encoder
            self._encoder_module_config = self.encoder_config
        elif isinstance(encoder, nn.Module):
            self.encoder = encoder
            self._encoder_module_type = "external"
        else:
            warnings.warn(
                f"{self.__class__.__name__} was initialized without an explicit encoder backbone.",
                stacklevel=3,
            )

        self.encoder_output_spec = self.encoder.output_spec if isinstance(self.encoder, BaseAutoencoderModule) else None
        self.core_spec = self.encoder_output_spec
        self.encoder_to_core_projection = None
        self.core_to_decoder_projection = None
        if self.config.latent_dim is not None and self.core_spec is not None:
            assert isinstance(self.core_spec, TensorSpec)
            assert self.core_spec.shape[-1] is not None
            # The core space is always defined by the model family, not the
            # backbone. When latent_dim is set, we adapt the backbone output into
            # that shared space and symmetrically adapt it back for the decoder.
            self.encoder_to_core_projection = self._build_projection(self.core_spec.shape[-1], self.config.latent_dim)
            self.core_to_decoder_projection = self._build_projection(self.config.latent_dim, self.core_spec.shape[-1])
            self.core_spec = self._build_projection_output_spec(self.core_spec, self.config.latent_dim)

        if self.core_spec is not None:
            self.validate_core_spec()

        self.decoder = None
        self.decoder_config = None
        if isinstance(decoder, str):
            decoder_class = get_module_class(decoder)
            decoder_config_class = decoder_class.config_class
            if decoder_config is None:
                raise ValueError(
                    f"{self.__class__.__name__} received built-in decoder={decoder!r} without `decoder_config`."
                )
            self.decoder_config = (
                decoder_config
                if isinstance(decoder_config, decoder_config_class)
                else decoder_config_class(**decoder_config)
            )
            self.decoder = decoder_class(
                config=self.decoder_config,
                input_spec=self.get_decoder_input_spec(),
            )
            self._decoder_module_type = decoder
            self._decoder_module_config = self.decoder_config
        elif isinstance(decoder, nn.Module):
            self.decoder = decoder
            self._decoder_module_type = "external"
        else:
            if self.encoder is None:
                warnings.warn(
                    f"{self.__class__.__name__} was initialized without an explicit decoder backbone because no "
                    "encoder backbone was available to derive it from.",
                    stacklevel=3,
                )
            elif isinstance(encoder, str):
                decoder_class = get_module_class(encoder)
                decoder_config_class = decoder_class.config_class
                self.decoder_config = (
                    encoder_config
                    if isinstance(encoder_config, decoder_config_class)
                    else decoder_config_class(**encoder_config)
                )
                self.validate_auto_inferred_decoder()
                # Reverse decoding is derived from the encoder reference spec so
                # the builder can reconstruct the original layer plan exactly.
                self.decoder = decoder_class(config=self.decoder_config, input_spec=self.sample_spec, reverse=True)
                self._decoder_module_type = encoder
                self._decoder_module_config = self.decoder_config
            else:
                raise ValueError(
                    f"{self.__class__.__name__} cannot infer decoder from encoder={encoder.__class__.__name__}. "
                    f"When `decoder=None`, the encoder must be a built-in or custom `{BaseAutoencoderModule.__name__}`."
                )

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

    def _build_projection(
        self,
        input_dim: int,
        output_dim: int,
    ) -> nn.Linear:
        return nn.Linear(input_dim, output_dim, bias=True)

    def _build_projection_output_spec(self, input_spec: DataSpec, output_dim: int) -> TensorSpec:
        if not isinstance(input_spec, TensorSpec):
            raise ValueError(f"{self.__class__.__name__} currently only supports TensorSpec projections.")
        return TensorSpec(shape=(*input_spec.shape[:-1], output_dim))

    def _require_backbone_module(self, module: nn.Module | None, name: str) -> nn.Module:
        if module is None:
            article = "an" if name[0].lower() in {"a", "e", "i", "o", "u"} else "a"
            raise RuntimeError(
                f"{self.__class__.__name__} does not have {article} {name} backbone. "
                f"Construct the model with `{name}=...` and the corresponding `{name}_config=...`."
            )
        return module

    def _encode_backbone(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._require_backbone_module(self.encoder, "encoder")(inputs)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into latent representations."""
        return self._require_backbone_module(self.encoder, "encoder")(inputs)

    def project_to_core(self, encoded: torch.Tensor) -> torch.Tensor:
        """Project encoder outputs into the core latent space when configured."""
        if self.encoder_to_core_projection is None:
            return encoded
        return self.encoder_to_core_projection(encoded)

    def project_from_core(self, latents: torch.Tensor) -> torch.Tensor:
        """Project core latents into decoder inputs when configured."""
        if self.core_to_decoder_projection is None:
            return latents
        return self.core_to_decoder_projection(latents)

    def validate_auto_inferred_decoder(self) -> None:
        if self.encoder_output_spec is None:
            return
        decoder_input_spec = self.get_decoder_input_spec()
        if decoder_input_spec.matches(self.encoder_output_spec):
            return
        raise ValueError(
            f"{self.__class__.__name__} cannot infer decoder from the encoder because the decoder input spec "
            f"{decoder_input_spec!r} does not match the encoder output spec {self.encoder_output_spec!r}. "
            "Provide an explicit decoder when the model decoder expects a different runtime input spec."
        )

    def prepare_decoder_inputs(self, latents: torch.Tensor) -> torch.Tensor:
        return self.project_from_core(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back into feature space."""
        return self._require_backbone_module(self.decoder, "decoder")(latents)

    def get_decoder_input_spec(self) -> DataSpec:
        if self.encoder_output_spec is not None:
            return self.encoder_output_spec
        return self.sample_spec

    def reconstruct(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(inputs=inputs, return_dict=True)
        return outputs.reconstruction

    def validate_core_spec(self) -> None:
        pass

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

    def get_core_trace(self, input_spec: DataSpec) -> list[PipelineTraceStep]:
        return [PipelineTraceStep(name="core", input_spec=input_spec, output_spec=input_spec)]

    def get_pipeline_trace(self) -> list[PipelineTraceStep]:
        trace: list[PipelineTraceStep] = []
        current_spec = self.sample_spec
        trace.append(PipelineTraceStep(name="sample", input_spec=current_spec, output_spec=current_spec))

        if isinstance(self.encoder, BaseAutoencoderModule):
            encoder_trace = self.encoder.get_trace()
            trace.append(
                PipelineTraceStep(
                    name=f"encoder[{self._encoder_module_type or self.encoder.__class__.__name__}]",
                    input_spec=self.encoder.input_spec,
                    output_spec=self.encoder.output_spec,
                    children=encoder_trace,
                )
            )
            current_spec = self.encoder.output_spec

        if self.encoder_to_core_projection is not None and self.config.latent_dim is not None:
            projected_spec = self._build_projection_output_spec(current_spec, self.config.latent_dim)
            trace.append(
                PipelineTraceStep(
                    name="encoder_to_core_projection",
                    input_spec=current_spec,
                    output_spec=projected_spec,
                )
            )
            current_spec = projected_spec

        core_trace = self.get_core_trace(current_spec)
        trace.extend(core_trace)
        if core_trace:
            current_spec = core_trace[-1].output_spec

        if self.core_to_decoder_projection is not None and self.encoder_output_spec is not None:
            assert isinstance(self.encoder_output_spec, TensorSpec)
            assert self.encoder_output_spec.shape[-1] is not None
            projected_spec = self._build_projection_output_spec(current_spec, self.encoder_output_spec.shape[-1])
            trace.append(
                PipelineTraceStep(
                    name="core_to_decoder_projection",
                    input_spec=current_spec,
                    output_spec=projected_spec,
                )
            )
            current_spec = projected_spec

        if isinstance(self.decoder, BaseAutoencoderModule):
            decoder_trace = self.decoder.get_trace()
            trace.append(
                PipelineTraceStep(
                    name=f"decoder[{self._decoder_module_type or self.decoder.__class__.__name__}]",
                    input_spec=self.decoder.input_spec,
                    output_spec=self.decoder.output_spec,
                    children=decoder_trace,
                )
            )

        return trace

    def forward(
        self,
        inputs: torch.Tensor,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> BaseAutoencoderOutput | tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs)
        core_inputs = self.project_to_core(encoded)
        latents = core_inputs
        decoder_inputs = self.prepare_decoder_inputs(latents)
        reconstruction = self.decode(decoder_inputs)

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
