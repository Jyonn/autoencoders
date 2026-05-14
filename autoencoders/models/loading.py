"""Model loading helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any


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
        for export_name in getattr(module, "__all__", [])
        if export_name.endswith("Model")
    ]
    if len(model_export_names) != 1:
        raise ValueError(
            f"Expected exactly one model export in module {module.__name__!r}, found {len(model_export_names)}."
        )
    model_class = getattr(module, model_export_names[0])
    if getattr(model_class, "config_class", None) is None:
        raise ValueError(f"Model export {model_class.__name__!r} in module {module.__name__!r} is missing config_class.")
    return model_class


def load_model(name: str, **kwargs: Any):
    """Construct a named autoencoder model from config kwargs."""

    model_class = get_model_class(name)
    config_class = model_class.config_class
    config = config_class(**kwargs)
    return model_class(config)
