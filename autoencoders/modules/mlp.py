"""Built-in MLP backbone module."""

from __future__ import annotations

from typing import Callable

from torch import nn

from ..data.base import DataSpec, TensorSpec
from ..function import get_activation_factory
from .base import (
    BaseAutoencoderModule,
    BaseAutoencoderModuleConfig,
    BaseModuleLayerBuilder,
    LayerBuilderList,
    ModuleTraceStep,
)

__all__ = [
    "MLPModule",
    "MLPModuleConfig",
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


class MLPLayerBuilder(BaseModuleLayerBuilder):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            first: bool,
            last: bool,
            activation: Callable[[], nn.Module],
            use_bias: bool,
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first = first
        self.last = last
        self.activation = activation
        self.use_bias = use_bias

    def reverse(self):
        self.in_dim, self.out_dim = self.out_dim, self.in_dim
        self.first, self.last = self.last, self.first

    def build(self) -> nn.Module:
        return MLPLayer(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            activation=None if self.last else self.activation(),
            use_bias=self.use_bias,
        )

    def infer_output_spec(self, input_spec: DataSpec) -> DataSpec:
        if not isinstance(input_spec, TensorSpec):
            raise ValueError(f"{self.__class__.__name__} expects TensorSpec inputs.")
        return TensorSpec(shape=(*input_spec.shape[:-1], self.out_dim))

    def get_trace_steps(self, input_spec: DataSpec) -> list[ModuleTraceStep]:
        linear_output_spec = self.infer_output_spec(input_spec)
        steps = [
            ModuleTraceStep(
                name=f"linear({self.in_dim}->{self.out_dim})",
                input_spec=input_spec,
                output_spec=linear_output_spec,
            )
        ]
        if not self.last:
            steps.append(
                ModuleTraceStep(
                    name=f"activation({self.activation().__class__.__name__})",
                    input_spec=linear_output_spec,
                    output_spec=linear_output_spec,
                )
            )
        return steps


class MLPLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: nn.Module | None,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.activation = activation

    def forward(self, inputs):  # type: ignore[override]
        outputs = self.linear(inputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

class MLPModule(BaseAutoencoderModule):
    """Reusable feed-forward backbone for vector inputs."""

    config_class = MLPModuleConfig
    config: MLPModuleConfig
    input_spec: TensorSpec
    output_spec: TensorSpec

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        reverse_requested = self.consume_reverse_flag()

        self._require_tensor_spec()
        dims = [self._resolve_input_dim(), *self.config.hidden_dims]
        self.output_spec = TensorSpec(shape=(*self.input_spec.shape[:-1], dims[-1]))

        self.builder_list = self._construct_builder_list(dims)
        if reverse_requested:
            self.builder_list.reverse()
            self.input_spec, self.output_spec = self.output_spec, self.input_spec
        self.network = self.builder_list.build()

    def forward(self, inputs):  # type: ignore[override]
        return self.network(inputs)

    def get_trace(self) -> list[ModuleTraceStep]:
        return self.builder_list.get_trace(self.input_spec)

    def _construct_builder_list(self, dims):
        builders = [
            MLPLayerBuilder(
                in_dim=in_dim,
                out_dim=out_dim,
                first=index == 0,
                last=index == len(dims) - 2,
                activation=get_activation_factory(self.config.activation),
                use_bias=self.config.use_bias,
            )
            for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        return LayerBuilderList(builders)

    def _require_tensor_spec(self):
        if not isinstance(self.input_spec, TensorSpec):
            raise ValueError(f"{self.__class__.__name__} expects a TensorSpec input.")
        if not self.input_spec.shape:
            raise ValueError(f"{self.__class__.__name__} requires at least one feature dimension.")
        if self.input_spec.shape[-1] is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a concrete final feature dimension to build the MLP."
            )

    def _resolve_input_dim(self) -> int:
        feature_dim = self.input_spec.shape[-1]
        assert feature_dim is not None
        return int(feature_dim)
