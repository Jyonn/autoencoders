"""PyTorch model utilities for the autoencoders library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


class PreTrainedAutoencoderModel(nn.Module):
    """Minimal pretrained model mixin for autoencoder-family models."""

    config_class = None
    base_model_prefix = "model"
    weights_name = "pytorch_model.bin"

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "PreTrainedAutoencoderModel":
        return cls(config, **kwargs)

    def get_serializable_module_specs(self) -> dict[str, dict[str, Any]]:
        return {}

    @classmethod
    def _load_serializable_module_specs(cls, load_path: Path) -> dict[str, dict[str, Any]]:
        module_specs: dict[str, dict[str, Any]] = {}
        for name in ("encoder", "decoder"):
            spec_path = load_path / f"{name}_module.json"
            if spec_path.exists():
                module_specs[name] = json.loads(spec_path.read_text(encoding="utf-8"))
        return module_specs

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_path)
        for name, spec in self.get_serializable_module_specs().items():
            spec_path = save_path / f"{name}_module.json"
            spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        weights_path = save_path / self.weights_name
        torch.save(self.state_dict(), weights_path)
        return weights_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        map_location: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "PreTrainedAutoencoderModel":
        load_path = Path(pretrained_model_name_or_path)
        config_kwargs = dict(kwargs)
        for name in ("encoder", "decoder", "encoder_config", "decoder_config"):
            config_kwargs.pop(name, None)
        config = cls.config_class.from_pretrained(load_path, **config_kwargs)
        module_specs = cls._load_serializable_module_specs(load_path) if load_path.is_dir() else {}

        init_kwargs = dict(kwargs)
        for name, spec in module_specs.items():
            module_type = spec["module_type"]
            if module_type == "external":
                if name not in init_kwargs:
                    raise ValueError(
                        f"{cls.__name__} was saved with an external {name} module. "
                        f"Please provide `{name}=...` when calling from_pretrained()."
                    )
                continue
            init_kwargs.setdefault(name, module_type)
            init_kwargs.setdefault(f"{name}_config", spec["module_config"])

        model = cls.from_config(config, **init_kwargs)

        weights_path = load_path / cls.weights_name if load_path.is_dir() else load_path
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=map_location)
            model.load_state_dict(state_dict)

        return model
