"""Model loading helpers."""

from __future__ import annotations

from importlib import import_module
import inspect
from typing import Any


MODEL_MODULES: dict[str, str] = {
    "ae": "autoencoders.models.ae.modeling_ae",
    "dae": "autoencoders.models.dae.modeling_dae",
    "cae": "autoencoders.models.cae.modeling_cae",
    "sae": "autoencoders.models.sae.modeling_sae",
    "topksae": "autoencoders.models.topksae.modeling_topksae",
    "klsae": "autoencoders.models.klsae.modeling_klsae",
    "vae": "autoencoders.models.vae.modeling_vae",
    "dvae": "autoencoders.models.dvae.modeling_dvae",
    "betavae": "autoencoders.models.betavae.modeling_betavae",
    "hvae": "autoencoders.models.hvae.modeling_hvae",
    "wae": "autoencoders.models.wae.modeling_wae",
    "aae": "autoencoders.models.aae.modeling_aae",
    "vqvae": "autoencoders.models.vqvae.modeling_vqvae",
    "fsq": "autoencoders.models.fsq.modeling_fsq",
    "pqvae": "autoencoders.models.pqvae.modeling_pqvae",
    "rqvae": "autoencoders.models.rqvae.modeling_rqvae",
}


def get_model_class(name: str):
    """Dynamically import and return a named autoencoder model class."""

    module_path = MODEL_MODULES.get(name)
    if module_path is None:
        available = ", ".join(repr(model_name) for model_name in MODEL_MODULES)
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
