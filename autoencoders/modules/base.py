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

    def __init__(self, **kwargs) -> None:
        config = kwargs.pop("config")
        input_spec = kwargs.pop("input_spec")
        latent_dim = kwargs.pop("latent_dim")
        reverse = kwargs.pop("reverse", False)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"{self.__class__.__name__} received unexpected keyword arguments: {unknown}")
        super().__init__()
        self.config = config
        self.input_spec = input_spec
        self.latent_dim = latent_dim
        self.reverse = reverse
        self.validate_input(self.input_spec)
        self.output_spec = self.infer_output_spec(self.input_spec)

    @abstractmethod
    def forward(self, inputs):  # type: ignore[override]
        """Run the backbone module."""

    @abstractmethod
    def validate_input(self, spec: DataSpec) -> None:
        """Raise if an input data spec is incompatible with this module."""

    @abstractmethod
    def infer_output_spec(self, spec: DataSpec) -> DataSpec:
        """Infer the structural output spec produced by this module."""
