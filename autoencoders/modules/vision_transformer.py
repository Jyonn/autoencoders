"""Built-in vision transformer backbone module for image inputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..data.base import DataSpec, TensorSpec
from .base import BaseAutoencoderModule, BaseAutoencoderModuleConfig, ModuleTraceStep

__all__ = [
    "VisionTransformerModule",
    "VisionTransformerModuleConfig",
]


def _normalize_pair(value: int | Sequence[int], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        return value, value

    items = tuple(int(item) for item in value)
    if len(items) != 2 or any(item <= 0 for item in items):
        raise ValueError(f"{name} must be a positive integer or a pair of positive integers.")
    return items


class VisionTransformerModuleConfig(BaseAutoencoderModuleConfig):
    """Configuration for the built-in vision transformer backbone."""

    model_type = "vision_transformer_module"

    patch_size: tuple[int, int]
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    use_bias: bool

    def __init__(
        self,
        patch_size: int | Sequence[int] = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        patch_size = _normalize_pair(patch_size, name="patch_size")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if dropout < 0:
            raise ValueError("dropout must be non-negative.")

        self.patch_size = patch_size
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.use_bias = bool(use_bias)
        super().__init__(**kwargs)


class VisionTransformerModule(BaseAutoencoderModule):
    """Reusable vision transformer backbone for HWC image specs."""

    config_class = VisionTransformerModuleConfig
    config: VisionTransformerModuleConfig
    input_spec: TensorSpec
    output_spec: TensorSpec

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._reference_image_spec = self._require_image_spec(self.input_spec)
        self.patch_height, self.patch_width = self.config.patch_size
        self.image_height, self.image_width, self.image_channels = self._resolve_image_shape(self._reference_image_spec)
        if self.image_height % self.patch_height != 0 or self.image_width % self.patch_width != 0:
            raise ValueError(
                f"{self.__class__.__name__} requires image height/width to be divisible by patch_size. "
                f"Received image shape {self._reference_image_spec.shape} and patch_size={self.config.patch_size}."
            )

        self.grid_height = self.image_height // self.patch_height
        self.grid_width = self.image_width // self.patch_width
        self.num_patches = self.grid_height * self.grid_width
        self.patch_dim = self.patch_height * self.patch_width * self.image_channels
        self.sequence_spec = TensorSpec(shape=(self.num_patches, self.config.hidden_dim))

        if self.reverse:
            self.input_spec = self.sequence_spec
            self.output_spec = self._reference_image_spec
        else:
            self.input_spec = self._reference_image_spec
            self.output_spec = self.sequence_spec

        self.patch_projection = nn.Linear(self.patch_dim, self.config.hidden_dim, bias=self.config.use_bias)
        self.transformer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.config.hidden_dim,
                    nhead=self.config.num_heads,
                    dim_feedforward=int(self.config.hidden_dim * self.config.mlp_ratio),
                    dropout=self.config.dropout,
                    batch_first=True,
                    activation="gelu",
                    bias=self.config.use_bias,
                    norm_first=True,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.output_projection = nn.Linear(self.config.hidden_dim, self.patch_dim, bias=self.config.use_bias)

    def forward(self, inputs):  # type: ignore[override]
        if self.reverse:
            if inputs.ndim != 3:
                raise ValueError(
                    f"{self.__class__.__name__} expects latent patch sequences with rank 3 when reverse=True, "
                    f"got {tuple(inputs.shape)}."
                )
            sequence = inputs
            for layer in self.transformer:
                sequence = layer(sequence)
            patch_values = self.output_projection(sequence)
            return self._unpatchify(patch_values)

        image_spec = self._require_image_spec(self.input_spec)
        self._validate_runtime_image(inputs, image_spec)
        patches = self._patchify(inputs)
        sequence = self.patch_projection(patches)
        for layer in self.transformer:
            sequence = layer(sequence)
        return sequence

    def get_trace(self) -> list[ModuleTraceStep]:
        if self.reverse:
            patch_spec = TensorSpec(shape=(self.num_patches, self.patch_dim))
            current_spec: DataSpec = self.input_spec
            steps: list[ModuleTraceStep] = []
            for index in range(self.config.num_layers):
                steps.append(
                    ModuleTraceStep(
                        name=(
                            f"transformer_layer[{index + 1}]"
                            f"(d={self.config.hidden_dim}, heads={self.config.num_heads})"
                        ),
                        input_spec=current_spec,
                        output_spec=current_spec,
                    )
                )
            steps.append(
                ModuleTraceStep(
                    name=f"linear({self.config.hidden_dim}->{self.patch_dim})",
                    input_spec=current_spec,
                    output_spec=patch_spec,
                )
            )
            steps.append(
                ModuleTraceStep(
                    name=f"unpatchify(p={self.config.patch_size})",
                    input_spec=patch_spec,
                    output_spec=self.output_spec,
                )
            )
            return steps

        patch_spec = TensorSpec(shape=(self.num_patches, self.patch_dim))
        steps = [
            ModuleTraceStep(
                name=f"patchify(p={self.config.patch_size})",
                input_spec=self.input_spec,
                output_spec=patch_spec,
            ),
            ModuleTraceStep(
                name=f"linear({self.patch_dim}->{self.config.hidden_dim})",
                input_spec=patch_spec,
                output_spec=self.output_spec,
            ),
        ]
        current_spec: DataSpec = self.output_spec
        for index in range(self.config.num_layers):
            steps.append(
                ModuleTraceStep(
                    name=(
                        f"transformer_layer[{index + 1}]"
                        f"(d={self.config.hidden_dim}, heads={self.config.num_heads})"
                    ),
                    input_spec=current_spec,
                    output_spec=current_spec,
                )
            )
        return steps

    def _patchify(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = inputs.shape
        assert height == self.image_height and width == self.image_width and channels == self.image_channels
        patches = inputs.reshape(
            batch_size,
            self.grid_height,
            self.patch_height,
            self.grid_width,
            self.patch_width,
            channels,
        )
        patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        return patches.reshape(batch_size, self.num_patches, self.patch_dim)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, patch_dim = patches.shape
        if num_patches != self.num_patches:
            raise ValueError(
                f"{self.__class__.__name__} expected {self.num_patches} patches, got {num_patches}."
            )
        if patch_dim != self.patch_dim:
            raise ValueError(
                f"{self.__class__.__name__} expected patch dimension {self.patch_dim}, got {patch_dim}."
            )
        images = patches.reshape(
            batch_size,
            self.grid_height,
            self.grid_width,
            self.patch_height,
            self.patch_width,
            self.image_channels,
        )
        images = images.permute(0, 1, 3, 2, 4, 5).contiguous()
        return images.reshape(batch_size, self.image_height, self.image_width, self.image_channels)

    @staticmethod
    def _require_image_spec(spec: DataSpec) -> TensorSpec:
        if not isinstance(spec, TensorSpec):
            raise ValueError("VisionTransformerModule expects TensorSpec inputs.")
        if len(spec.shape) != 3:
            raise ValueError(
                f"VisionTransformerModule expects image-like TensorSpec(shape=(H, W, C)), got {spec.shape}."
            )
        return spec

    @staticmethod
    def _resolve_image_shape(spec: TensorSpec) -> tuple[int, int, int]:
        if any(dimension is None for dimension in spec.shape):
            raise ValueError(
                "VisionTransformerModule requires concrete image dimensions to build patch projections."
            )
        height, width, channels = spec.shape
        assert height is not None and width is not None and channels is not None
        return int(height), int(width), int(channels)

    def _validate_runtime_image(self, inputs: torch.Tensor, spec: TensorSpec) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                f"{self.__class__.__name__} expects batched image tensors with rank 4, got {tuple(inputs.shape)}."
            )
        _, height, width, channels = inputs.shape
        expected_height, expected_width, expected_channels = self._resolve_image_shape(spec)
        if (height, width, channels) != (expected_height, expected_width, expected_channels):
            raise ValueError(
                f"{self.__class__.__name__} expected runtime image shape "
                f"(*, {expected_height}, {expected_width}, {expected_channels}), "
                f"got {tuple(inputs.shape)}."
            )
