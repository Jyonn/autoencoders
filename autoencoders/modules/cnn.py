"""Built-in CNN backbone module for image-like tensor inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
    "CNNModule",
    "CNNModuleConfig",
]


def _normalize_spatial_parameter(
    value: int | Sequence[int] | Sequence[Sequence[int]],
    *,
    count: int,
    name: str,
) -> list[tuple[int, int]]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        return [(value, value) for _ in range(count)]

    values = list(value)
    if len(values) != count:
        raise ValueError(f"{name} must provide exactly {count} values.")

    normalized: list[tuple[int, int]] = []
    for index, item in enumerate(values):
        if isinstance(item, int):
            if item <= 0:
                raise ValueError(f"{name}[{index}] must be positive.")
            normalized.append((item, item))
            continue

        pair = tuple(item)
        if len(pair) != 2 or any(component <= 0 for component in pair):
            raise ValueError(f"{name}[{index}] must be a positive integer or a pair of positive integers.")
        normalized.append((int(pair[0]), int(pair[1])))
    return normalized


def _infer_conv_output_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
    output_size = ((size + 2 * padding - kernel_size) // stride) + 1
    if output_size <= 0:
        raise ValueError(
            "CNN layer configuration produces a non-positive spatial dimension. "
            f"Received size={size}, kernel_size={kernel_size}, stride={stride}, padding={padding}."
        )
    return output_size


def _infer_output_padding(
    *,
    input_size: int,
    target_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> int:
    base_size = (input_size - 1) * stride - (2 * padding) + kernel_size
    output_padding = target_size - base_size
    if output_padding < 0 or output_padding >= stride:
        raise ValueError(
            "CNN reverse configuration requires unsupported output_padding. "
            f"Got input_size={input_size}, target_size={target_size}, kernel_size={kernel_size}, "
            f"stride={stride}, padding={padding}, inferred output_padding={output_padding}."
        )
    return output_padding


class CNNModuleConfig(BaseAutoencoderModuleConfig):
    """Configuration for the built-in CNN backbone."""

    model_type = "cnn_module"

    channels: list[int]
    kernel_sizes: list[tuple[int, int]]
    strides: list[tuple[int, int]]
    paddings: list[tuple[int, int]]
    activation: str
    use_bias: bool
    transpose: bool

    def __init__(
        self,
        channels: list[int] | None = None,
        kernel_sizes: int | Sequence[int] | Sequence[Sequence[int]] = 4,
        strides: int | Sequence[int] | Sequence[Sequence[int]] = 2,
        paddings: int | Sequence[int] | Sequence[Sequence[int]] = 1,
        activation: str = "relu",
        use_bias: bool = True,
        transpose: bool = False,
        **kwargs,
    ) -> None:
        channels = [] if channels is None else [int(channel) for channel in channels]
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain at least one positive integer.")
        if activation not in {"relu", "gelu", "silu", "tanh"}:
            raise ValueError("activation must be one of: 'relu', 'gelu', 'silu', 'tanh'.")

        self.channels = channels
        self.kernel_sizes = _normalize_spatial_parameter(kernel_sizes, count=len(channels), name="kernel_sizes")
        self.strides = _normalize_spatial_parameter(strides, count=len(channels), name="strides")
        self.paddings = _normalize_spatial_parameter(paddings, count=len(channels), name="paddings")
        self.activation = activation
        self.use_bias = use_bias
        self.transpose = bool(transpose)
        super().__init__(**kwargs)


@dataclass(frozen=True)
class _CNNLayerSpec:
    input_spec: TensorSpec
    output_spec: TensorSpec
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    output_padding: tuple[int, int] = (0, 0)
    transpose: bool = False
    last: bool = False


class CNNLayerBuilder(BaseModuleLayerBuilder):
    def __init__(
        self,
        *,
        layer_spec: _CNNLayerSpec,
        activation: Callable[[], nn.Module],
        use_bias: bool,
    ) -> None:
        self.layer_spec = layer_spec
        self.activation = activation
        self.use_bias = use_bias

    def reverse(self):
        spec = self.layer_spec
        input_height, input_width, _ = spec.input_spec.shape
        output_height, output_width, _ = spec.output_spec.shape
        assert input_height is not None and input_width is not None
        assert output_height is not None and output_width is not None
        output_padding = (
            _infer_output_padding(
                input_size=int(output_height),
                target_size=int(input_height),
                kernel_size=spec.kernel_size[0],
                stride=spec.stride[0],
                padding=spec.padding[0],
            ),
            _infer_output_padding(
                input_size=int(output_width),
                target_size=int(input_width),
                kernel_size=spec.kernel_size[1],
                stride=spec.stride[1],
                padding=spec.padding[1],
            ),
        )
        self.layer_spec = _CNNLayerSpec(
            input_spec=spec.output_spec,
            output_spec=spec.input_spec,
            in_channels=spec.out_channels,
            out_channels=spec.in_channels,
            kernel_size=spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding,
            output_padding=output_padding,
            transpose=not spec.transpose,
            last=spec.last,
        )
        return self

    def build(self) -> nn.Module:
        return CNNLayer(
            layer_spec=self.layer_spec,
            activation=None if self.layer_spec.last else self.activation(),
            use_bias=self.use_bias,
        )

    def infer_output_spec(self, input_spec: DataSpec) -> DataSpec:
        if not isinstance(input_spec, TensorSpec):
            raise ValueError(f"{self.__class__.__name__} expects TensorSpec inputs.")
        if not self.layer_spec.input_spec.matches(input_spec):
            raise ValueError(
                f"{self.__class__.__name__} expected input spec {self.layer_spec.input_spec}, "
                f"but received {input_spec}."
            )
        return self.layer_spec.output_spec

    def get_trace_steps(self, input_spec: DataSpec) -> list[ModuleTraceStep]:
        output_spec = self.infer_output_spec(input_spec)
        kind = "convtranspose2d" if self.layer_spec.transpose else "conv2d"
        steps = [
            ModuleTraceStep(
                name=(
                    f"{kind}({self.layer_spec.in_channels}->{self.layer_spec.out_channels}, "
                    f"k={self.layer_spec.kernel_size}, s={self.layer_spec.stride}, p={self.layer_spec.padding})"
                ),
                input_spec=input_spec,
                output_spec=output_spec,
            )
        ]
        if not self.layer_spec.last:
            steps.append(
                ModuleTraceStep(
                    name=f"activation({self.activation().__class__.__name__})",
                    input_spec=output_spec,
                    output_spec=output_spec,
                )
            )
        return steps


class CNNLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_spec: _CNNLayerSpec,
        activation: nn.Module | None,
        use_bias: bool,
    ) -> None:
        super().__init__()
        if layer_spec.transpose:
            self.layer = nn.ConvTranspose2d(
                layer_spec.in_channels,
                layer_spec.out_channels,
                kernel_size=layer_spec.kernel_size,
                stride=layer_spec.stride,
                padding=layer_spec.padding,
                output_padding=layer_spec.output_padding,
                bias=use_bias,
            )
        else:
            self.layer = nn.Conv2d(
                layer_spec.in_channels,
                layer_spec.out_channels,
                kernel_size=layer_spec.kernel_size,
                stride=layer_spec.stride,
                padding=layer_spec.padding,
                bias=use_bias,
            )
        self.activation = activation

    def forward(self, inputs):  # type: ignore[override]
        outputs = self.layer(inputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class CNNModule(BaseAutoencoderModule):
    """Reusable CNN backbone for image-like HWC inputs."""

    config_class = CNNModuleConfig
    config: CNNModuleConfig
    input_spec: TensorSpec
    output_spec: TensorSpec

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        reverse_requested = self.consume_reverse_flag()
        self._require_image_spec()

        # Build the reference plan from the provided HWC sample spec. The
        # config can explicitly request transposed layers, while `reverse=True`
        # still flips the whole plan when deriving a decoder from an encoder.
        builders = self._construct_builders()
        self.output_spec = builders.builders[-1].layer_spec.output_spec
        self.builder_list = builders
        if reverse_requested:
            self.builder_list.reverse()
            self.input_spec = self.builder_list.builders[0].layer_spec.input_spec
            self.output_spec = self.builder_list.builders[-1].layer_spec.output_spec
        self.network = self.builder_list.build()

    def forward(self, inputs):  # type: ignore[override]
        if inputs.ndim != 4:
            raise ValueError(f"{self.__class__.__name__} expects batched image tensors with rank 4, got {tuple(inputs.shape)}.")
        # Public specs stay HWC because they are easier to reason about beside
        # dataset specs, while the actual convolution stack still runs in NCHW.
        nchw_inputs = inputs.movedim(-1, 1).contiguous()
        nchw_outputs = self.network(nchw_inputs)
        return nchw_outputs.movedim(1, -1).contiguous()

    def get_trace(self) -> list[ModuleTraceStep]:
        return self.builder_list.get_trace(self.input_spec)

    def _construct_builders(self) -> LayerBuilderList:
        input_height, input_width, input_channels = self.input_spec.shape
        assert input_height is not None and input_width is not None and input_channels is not None

        activation = get_activation_factory(self.config.activation)
        builders: list[CNNLayerBuilder] = []
        current_spec = self.input_spec
        current_channels = int(input_channels)
        current_height = int(input_height)
        current_width = int(input_width)

        for index, out_channels in enumerate(self.config.channels):
            kernel_size = self.config.kernel_sizes[index]
            stride = self.config.strides[index]
            padding = self.config.paddings[index]
            if self.config.transpose:
                next_height = (current_height - 1) * stride[0] - (2 * padding[0]) + kernel_size[0]
                next_width = (current_width - 1) * stride[1] - (2 * padding[1]) + kernel_size[1]
            else:
                next_height = _infer_conv_output_size(current_height, kernel_size[0], stride[0], padding[0])
                next_width = _infer_conv_output_size(current_width, kernel_size[1], stride[1], padding[1])
            next_spec = TensorSpec(shape=(next_height, next_width, out_channels))
            builders.append(
                CNNLayerBuilder(
                    layer_spec=_CNNLayerSpec(
                        input_spec=current_spec,
                        output_spec=next_spec,
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        transpose=self.config.transpose,
                        last=index == len(self.config.channels) - 1,
                    ),
                    activation=activation,
                    use_bias=self.config.use_bias,
                )
            )
            current_spec = next_spec
            current_channels = out_channels
            current_height = next_height
            current_width = next_width
        return LayerBuilderList(builders)

    def _require_image_spec(self) -> None:
        if not isinstance(self.input_spec, TensorSpec):
            raise ValueError(f"{self.__class__.__name__} expects a TensorSpec input.")
        if len(self.input_spec.shape) != 3:
            raise ValueError(
                f"{self.__class__.__name__} expects image-like sample specs shaped as (height, width, channels), "
                f"got {self.input_spec.shape}."
            )
        if any(dimension is None for dimension in self.input_spec.shape):
            raise ValueError(
                f"{self.__class__.__name__} requires concrete spatial dimensions and channels, got {self.input_spec.shape}."
            )
