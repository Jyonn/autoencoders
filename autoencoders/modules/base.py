"""Base classes for reusable encoder/decoder backbone modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from ..configuration_utils import PretrainedConfig


class BaseAutoencoderModuleConfig(PretrainedConfig):
    """Base configuration shared by built-in backbone modules."""

    model_type = "base_module"


class BaseAutoencoderModule(nn.Module, ABC):
    """Base class for reusable backbone modules."""

    config_class = BaseAutoencoderModuleConfig

    def __init__(
        self,
        config: BaseAutoencoderModuleConfig,
        *,
        input_dim: int,
        latent_dim: int,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reverse = reverse

    @abstractmethod
    def forward(self, inputs):  # type: ignore[override]
        """Run the backbone module."""
