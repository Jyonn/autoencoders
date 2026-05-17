"""Base classes for reusable encoder/decoder backbone modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn

from ..configuration_utils import PretrainedConfig
from ..data.base import DataSpec


class BaseAutoencoderModuleConfig(PretrainedConfig):
    """Base configuration shared by built-in backbone modules."""

    model_type = "base_module"


@dataclass(frozen=True)
class ModuleTraceStep:
    """A single shape transition inside a backbone module."""

    name: str
    input_spec: DataSpec
    output_spec: DataSpec


class BaseModuleLayerBuilder(ABC):
    @abstractmethod
    def reverse(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def infer_output_spec(self, input_spec: DataSpec) -> DataSpec:
        pass

    @abstractmethod
    def get_trace_steps(self, input_spec: DataSpec) -> list[ModuleTraceStep]:
        pass


class LayerBuilderList:
    builders: list[BaseModuleLayerBuilder]

    def __init__(self, builders: list[BaseModuleLayerBuilder]):
        self.builders = builders

    def reverse(self):
        self.builders.reverse()
        for layer in self.builders:
            layer.reverse()
        return self

    def build(self):
        return nn.Sequential(*(layer.build() for layer in self.builders))

    def get_trace(self, input_spec: DataSpec) -> list[ModuleTraceStep]:
        steps: list[ModuleTraceStep] = []
        current_spec = input_spec
        for builder in self.builders:
            builder_steps = builder.get_trace_steps(current_spec)
            steps.extend(builder_steps)
            if builder_steps:
                current_spec = builder_steps[-1].output_spec
            else:
                current_spec = builder.infer_output_spec(current_spec)
        return steps


class BaseAutoencoderModule(nn.Module, ABC):
    """Base class for reusable backbone modules."""

    config_class = BaseAutoencoderModuleConfig
    config: BaseAutoencoderModuleConfig
    input_spec: DataSpec
    output_spec: DataSpec

    def __init__(
        self,
        config: BaseAutoencoderModuleConfig,
        input_spec: DataSpec,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.reverse = bool(reverse)
        self.input_spec = input_spec

    @abstractmethod
    def forward(self, inputs):  # type: ignore[override]
        """Run the backbone module."""

    def get_trace(self) -> list[ModuleTraceStep]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement trace inspection.")

    def build_reversed(self):
        return self.__class__(config=self.config, input_spec=self.input_spec, reverse=True)
