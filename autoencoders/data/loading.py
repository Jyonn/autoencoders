"""Dataset loading helpers."""

from __future__ import annotations

from glob import glob
import importlib
from pathlib import Path
from typing import Any

from .base import CachedDataset

_SKIP_DATASET_MODULES = {"__init__", "base", "clip", "embeddings", "loading", "text"}


def _dataset_module_paths() -> list[Path]:
    package_dir = Path(__file__).resolve().parent
    module_paths = [Path(path) for path in glob(str(package_dir / "*.py"))]
    return sorted(path for path in module_paths if path.stem not in _SKIP_DATASET_MODULES)


def _dataset_class_from_module(module) -> type[CachedDataset]:
    candidates = [
        value
        for value in module.__dict__.values()
        if isinstance(value, type)
        and issubclass(value, CachedDataset)
        and value is not CachedDataset
        and value.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one dataset class in module {module.__name__!r}, found {len(candidates)}."
        )
    return candidates[0]


def get_dataset_modules() -> dict[str, str]:
    """Return the discovered dataset modules keyed by dataset name."""

    modules: dict[str, str] = {}
    for module_path in _dataset_module_paths():
        module_name = f"{__package__}.{module_path.stem}"
        module = importlib.import_module(module_name)
        dataset_class = _dataset_class_from_module(module)
        modules[dataset_class.dataset_name] = module_name
    return modules


def get_dataset_class(name: str) -> type[CachedDataset]:
    """Return the dataset class registered under a given dataset name."""

    module_name = get_dataset_modules().get(name)
    if module_name is None:
        available = ", ".join(repr(dataset_name) for dataset_name in sorted(get_dataset_modules()))
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}.")
    module = importlib.import_module(module_name)
    return _dataset_class_from_module(module)


def load_dataset(name: str, config=None, **kwargs: Any) -> CachedDataset:
    """Load a named dataset, downloading and caching it on demand."""

    dataset_class = get_dataset_class(name)
    if config is None:
        config = dataset_class.config_class(**kwargs)
    elif kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"load_dataset() received both `config` and extra keyword arguments: {unknown}")
    return dataset_class(config)
