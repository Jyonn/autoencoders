"""Built-in MLP backbone module."""

from __future__ import annotations

from typing import Callable

from torch import nn

from .base import BaseAutoencoderModule, BaseAutoencoderModuleConfig

__all__ = ["MLPModule", "MLPModuleConfig"]


class MLPModuleConfig(BaseAutoencoderModuleConfig):
    """Configuration for the built-in MLP backbone."""

    model_type = "mlp_module"

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
        super().__init__(
            hidden_dims=hidden_dims,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class MLPModule(BaseAutoencoderModule):
    """Reusable feed-forward backbone for vector inputs."""

    config_class = MLPModuleConfig

    def __init__(
        self,
        config: MLPModuleConfig,
        *,
        input_dim: int,
        latent_dim: int,
        reverse: bool = False,
    ) -> None:
        super().__init__(config, input_dim=input_dim, latent_dim=latent_dim, reverse=reverse)
        hidden_dims = list(reversed(config.hidden_dims)) if reverse else list(config.hidden_dims)
        dims = [input_dim, *hidden_dims, latent_dim]
        self.network = self._build_mlp(dims)

    def forward(self, inputs):  # type: ignore[override]
        return self.network(inputs)

    def _build_mlp(self, dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        activation_factory = self._get_activation_factory()

        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim, bias=self.config.use_bias))
            is_last_layer = index == len(dims) - 2
            if not is_last_layer:
                layers.append(activation_factory())

        return nn.Sequential(*layers)

    def _get_activation_factory(self) -> Callable[[], nn.Module]:
        activations: dict[str, Callable[[], nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        return activations[self.config.activation]
