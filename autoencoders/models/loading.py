"""Model loading helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from ..data.base import TensorSpec

MODELS_ROOT = Path(__file__).resolve().parent


def _discover_model_modules() -> dict[str, str]:
    module_map: dict[str, str] = {}
    for modeling_path in sorted(MODELS_ROOT.glob("*/modeling_*.py")):
        namespace = modeling_path.parent.name
        if namespace == "base":
            continue
        module_map[namespace] = f"autoencoders.models.{namespace}"
    return module_map


@lru_cache(maxsize=1)
def get_model_modules() -> dict[str, str]:
    return _discover_model_modules()


def get_model_class(name: str):
    """Dynamically import and return a named autoencoder model class."""

    module_path = get_model_modules().get(name)
    if module_path is None:
        available = ", ".join(repr(model_name) for model_name in get_model_modules())
        raise ValueError(f"Unknown model {name!r}. Available models: {available}.")

    module = import_module(module_path)
    model_export_names = [
        export_name
        for export_name in module.__all__
        if export_name.endswith("Model")
    ]
    if len(model_export_names) != 1:
        raise ValueError(
            f"Expected exactly one model export in module {module.__name__!r}, found {len(model_export_names)}."
        )
    return getattr(module, model_export_names[0])


def load_model(name: str, **kwargs: Any):
    """Construct a named autoencoder model from config kwargs."""

    init_kwargs: dict[str, Any] = {}
    for init_key in ("encoder", "decoder", "encoder_config", "decoder_config", "sample_spec"):
        if init_key in kwargs:
            init_kwargs[init_key] = kwargs.pop(init_key)

    sample_spec = init_kwargs.get("sample_spec")
    if "input_dim" not in kwargs and sample_spec is not None:
        if not isinstance(sample_spec, TensorSpec) or not sample_spec.shape:
            raise ValueError(
                "load_model() can only infer `input_dim` from a TensorSpec with a concrete final dimension."
            )
        input_dim = sample_spec.shape[-1]
        if input_dim is None:
            raise ValueError(
                "load_model() requires a concrete final dimension in `sample_spec` to infer `input_dim`."
            )
        kwargs["input_dim"] = int(input_dim)

    model_class = get_model_class(name)
    config_class = model_class.config_class
    config = config_class(**kwargs)
    return model_class(config=config, **init_kwargs)
