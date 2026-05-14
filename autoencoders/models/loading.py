"""Model loading helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import inspect
from pathlib import Path
from typing import Any


MODELS_ROOT = Path(__file__).resolve().parent


def _discover_model_modules() -> dict[str, str]:
    module_map: dict[str, str] = {}
    for modeling_path in sorted(MODELS_ROOT.glob("*/modeling_*.py")):
        namespace = modeling_path.parent.name
        if namespace == "base":
            continue
        module_map[namespace] = f"autoencoders.models.{namespace}.{modeling_path.stem}"
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
    model_classes = [
        value
        for _, value in inspect.getmembers(module, inspect.isclass)
        if value.__module__ == module.__name__
        and value.__name__.endswith("Model")
        and getattr(value, "config_class", None) is not None
    ]
    if len(model_classes) != 1:
        raise ValueError(
            f"Expected exactly one model class in module {module.__name__!r}, found {len(model_classes)}."
        )
    return model_classes[0]


def load_model(name: str, **kwargs: Any):
    """Construct a named autoencoder model from config kwargs."""

    model_class = get_model_class(name)
    config_class = model_class.config_class
    config = config_class(**kwargs)
    return model_class(config)
