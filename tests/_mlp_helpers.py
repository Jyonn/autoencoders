"""Test-only helpers for wiring built-in MLP backbones."""

from __future__ import annotations


def build_mlp_backbone_kwargs(
    hidden_dims: list[int],
    input_dim: int | None = None,
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
        ([*list(reversed(hidden_dims[:-1])), int(input_dim)] if input_dim is not None else None)
        if decoder_hidden_dims is None
        else list(decoder_hidden_dims)
    )
    if effective_decoder_hidden_dims is None:
        raise ValueError("input_dim is required when decoder_hidden_dims is not provided.")
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
        input_dim=config.input_dim,
        activation=getattr(config, "activation", "relu"),
        use_bias=getattr(config, "use_bias", True),
        decoder_hidden_dims=(
            None
            if getattr(config, "decoder_hidden_dims", None) is None
            else list(config.decoder_hidden_dims)
        ),
    )
