"""Dynamic loader for built-in backbone modules."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path

from .base import BaseAutoencoderModule


def _normalize_module_name(module_name: str) -> str:
    return module_name.replace("-", "").replace("_", "").lower()


@lru_cache(maxsize=1)
def get_module_modules() -> dict[str, str]:
    modules_dir = Path(__file__).resolve().parent
    discovered: dict[str, str] = {}
    for module_path in sorted(modules_dir.glob("*.py")):
        if module_path.stem in {"__init__", "base", "loading"}:
            continue
        discovered[_normalize_module_name(module_path.stem)] = f"autoencoders.modules.{module_path.stem}"
    return discovered


def get_module_class(module_name: str) -> type[BaseAutoencoderModule]:
    normalized_name = _normalize_module_name(module_name)
    module_path = get_module_modules()[normalized_name]
    module = import_module(module_path)
    module_names = [name for name in module.__all__ if name.endswith("Module")]
    return getattr(module, module_names[0])
