"""Built-in MLP backbone module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from ..data.base import DataSpec, TensorSpec
from .base import BaseAutoencoderModule, BaseAutoencoderModuleConfig

__all__ = [
    "MLPModule",
    "MLPModuleConfig",
    "build_mlp_backbone_kwargs",
    "build_mlp_backbone_kwargs_from_model_config",
]


class MLPModuleConfig(BaseAutoencoderModuleConfig):
    """Configuration for the built-in MLP backbone."""

    model_type = "mlp_module"

    hidden_dims: list[int]
    activation: str
    use_bias: bool

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        hidden_dims = [] if hidden_dims is None else list(hidden_dims)
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if activation not in {"relu", "gelu", "silu", "tanh"}:
            raise ValueError("activation must be one of: 'relu', 'gelu', 'silu', 'tanh'.")
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.use_bias = use_bias
        super().__init__(**kwargs)


@dataclass(frozen=True)
class _MLPBuilder:
    in_dim: int
    out_dim: int
    first: bool
    last: bool
    activation: str
    use_bias: bool

    def reverse(self) -> "_MLPBuilder":
        return _MLPBuilder(
            in_dim=self.out_dim,
            out_dim=self.in_dim,
            first=self.last,
            last=self.first,
            activation=self.activation,
            use_bias=self.use_bias,
        )

    def build(self) -> list[nn.Module]:
        layers: list[nn.Module] = [nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)]
        if not self.last:
            layers.append(_get_activation_factory(self.activation)())
        return layers


@dataclass(frozen=True)
class _MLPBuilderList:
    builders: tuple[_MLPBuilder, ...]

    @classmethod
    def from_dims(
        cls,
        dims: list[int],
        *,
        activation: str,
        use_bias: bool,
    ) -> "_MLPBuilderList":
        builders = tuple(
            _MLPBuilder(
                in_dim=in_dim,
                out_dim=out_dim,
                first=index == 0,
                last=index == len(dims) - 2,
                activation=activation,
                use_bias=use_bias,
            )
            for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
        )
        return cls(builders)

    def reverse(self) -> "_MLPBuilderList":
        return _MLPBuilderList(tuple(builder.reverse() for builder in reversed(self.builders)))

    def build(self) -> nn.Sequential:
        if not self.builders:
            return nn.Sequential(nn.Identity())
        layers: list[nn.Module] = []
        for builder in self.builders:
            layers.extend(builder.build())
        return nn.Sequential(*layers)

    @property
    def output_dim(self) -> int | None:
        if not self.builders:
            return None
        return self.builders[-1].out_dim


def _get_activation_factory(activation: str) -> Callable[[], nn.Module]:
    activations: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    return activations[activation]


class MLPModule(BaseAutoencoderModule):
    """Reusable feed-forward backbone for vector inputs."""

    config_class = MLPModuleConfig
    config: MLPModuleConfig
    builder_list: _MLPBuilderList

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        input_dim = self._resolve_input_dim()
        dims = [input_dim, *self.config.hidden_dims]
        self.builder_list = _MLPBuilderList.from_dims(
            dims,
            activation=self.config.activation,
            use_bias=self.config.use_bias,
        )
        if self.reverse:
            self.builder_list = self.builder_list.reverse()
        self.network = self.builder_list.build()

    def forward(self, inputs):  # type: ignore[override]
        return self.network(inputs)

    def infer_output_spec(self) -> TensorSpec:
        spec = self.input_spec
        if not isinstance(spec, TensorSpec):
            raise ValueError(f"{self.__class__.__name__} expects a TensorSpec input.")
        if not spec.shape:
            raise ValueError(f"{self.__class__.__name__} requires at least one feature dimension.")
        if spec.shape[-1] is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a concrete final feature dimension to build the MLP."
            )
        return TensorSpec(shape=(*spec.shape[:-1], self._resolve_output_feature_dim()))

    def build_reversed(self, input_spec: DataSpec) -> "MLPModule":
        reversed_module = MLPModule(
            config=self.config,
            input_spec=input_spec,
        )
        reversed_module.builder_list = self.builder_list.reverse()
        reversed_module.network = reversed_module.builder_list.build()
        reversed_output_dim = reversed_module.builder_list.output_dim
        if reversed_output_dim is None:
            reversed_output_dim = reversed_module._resolve_input_dim()
        reversed_module.output_spec = TensorSpec(shape=(*reversed_module.input_spec.shape[:-1], reversed_output_dim))
        return reversed_module

    def _resolve_input_dim(self) -> int:
        spec = self.input_spec
        assert isinstance(spec, TensorSpec)
        feature_dim = spec.shape[-1]
        assert feature_dim is not None
        return int(feature_dim)

    def _resolve_output_feature_dim(self) -> int:
        if hasattr(self, "builder_list") and self.builder_list.output_dim is not None:
            return self.builder_list.output_dim
        if self.config.hidden_dims:
            if self.reverse:
                return self.config.hidden_dims[0]
            return self.config.hidden_dims[-1]
        return self._resolve_input_dim()


def build_mlp_backbone_kwargs(
    hidden_dims: list[int],
    activation: str = "relu",
    use_bias: bool = True,
    decoder_hidden_dims: list[int] | None = None,
) -> dict[str, object]:
    encoder_config = {
        "hidden_dims": list(hidden_dims),
        "activation": activation,
        "use_bias": use_bias,
    }
    effective_decoder_hidden_dims = (
        list(reversed(hidden_dims))
        if decoder_hidden_dims is None
        else list(decoder_hidden_dims)
    )
    decoder_config = {
        "hidden_dims": effective_decoder_hidden_dims,
        "activation": activation,
        "use_bias": use_bias,
    }
    return {
        "encoder": "mlp",
        "decoder": "mlp",
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
    }


def build_mlp_backbone_kwargs_from_model_config(config) -> dict[str, object]:
    return build_mlp_backbone_kwargs(
        hidden_dims=list(config.hidden_dims),
        activation=getattr(config, "activation", "relu"),
        use_bias=getattr(config, "use_bias", True),
        decoder_hidden_dims=(
            None
            if getattr(config, "decoder_hidden_dims", None) is None
            else list(config.decoder_hidden_dims)
        ),
    )
