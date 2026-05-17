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
    input_spec: DataSpec
    output_spec: DataSpec
    reverse: bool

    def __init__(
        self,
        config: BaseAutoencoderModuleConfig,
        input_spec: DataSpec,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_spec = input_spec
        self.reverse = bool(reverse)
        self.output_spec = self.infer_output_spec()

    @abstractmethod
    def forward(self, inputs):  # type: ignore[override]
        """Run the backbone module."""

    @abstractmethod
    def infer_output_spec(self) -> DataSpec:
        """Validate an input spec and infer the structural output spec produced by this module."""
