"""Built-in MLP backbone module."""

from __future__ import annotations

from typing import Callable

from torch import nn

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


class MLPModule(BaseAutoencoderModule):
    """Reusable feed-forward backbone for vector inputs."""

    config_class = MLPModuleConfig
    config: MLPModuleConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        hidden_dims = (
            list(reversed(self.config.hidden_dims))
            if self.reverse
            else list(self.config.hidden_dims)
        )
        dims = [self.input_dim, *hidden_dims, self.latent_dim]
        self.network = self._build_mlp(dims)

    def forward(self, inputs):  # type: ignore[override]
        return self.network(inputs)

    def get_output_dim(self) -> int:
        return self.latent_dim

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
