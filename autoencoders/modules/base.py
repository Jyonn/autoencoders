"""Base classes for reusable encoder/decoder backbone modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from ..configuration_utils import PretrainedConfig
from ..data.base import DataSpec


class BaseAutoencoderModuleConfig(PretrainedConfig):
    """Base configuration shared by built-in backbone modules."""

    model_type = "base_module"


class BaseAutoencoderModule(nn.Module, ABC):
    """Base class for reusable backbone modules."""

    config_class = BaseAutoencoderModuleConfig
    config: BaseAutoencoderModuleConfig
    reference_input_spec: DataSpec

    def __init__(
        self,
        config: BaseAutoencoderModuleConfig,
        input_spec: DataSpec,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.reference_input_spec = input_spec
        self.reverse = bool(reverse)
        self.input_spec = self.infer_input_spec()
        self.output_spec = self.infer_output_spec()

    @abstractmethod
    def forward(self, inputs):  # type: ignore[override]
        """Run the backbone module."""

    def infer_input_spec(self) -> DataSpec:
        return self.reference_input_spec

    @abstractmethod
    def infer_output_spec(self) -> DataSpec:
        """Validate an input spec and infer the structural output spec produced by this module."""

    def build_reversed(self) -> "BaseAutoencoderModule":
        return self.__class__(config=self.config, input_spec=self.reference_input_spec, reverse=True)
