"""Model loading helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

from ..data.base import TensorSpec

if TYPE_CHECKING:
    from .base.modeling_base import BaseAutoencoderModel

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


def get_model_class(name: str) -> Type["BaseAutoencoderModel"]:
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
    if sample_spec is None and "input_dim" in kwargs:
        input_dim = kwargs.pop("input_dim")
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("load_model() requires `input_dim` to be a positive integer when used as a fallback.")
        quantized_models = {"vqvae", "gumbelvq", "fsq", "rfsq", "pqvae", "rqvae", "vqvae2"}
        sample_spec = TensorSpec(shape=((None, input_dim) if name in quantized_models else (input_dim,)))
        init_kwargs["sample_spec"] = sample_spec
    if sample_spec is None:
        raise ValueError("load_model() requires `sample_spec`.")

    model_class = get_model_class(name)
    config_class = model_class.config_class
    config = kwargs.pop("config", None)
    if config is None:
        config = config_class(**kwargs)
    elif kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"load_model() received both `config` and extra keyword arguments: {unknown}")
    return model_class(config=config, **init_kwargs)
