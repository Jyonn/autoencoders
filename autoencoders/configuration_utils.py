"""Utilities for model configuration objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PretrainedConfig:
    """A lightweight configuration base inspired by the transformers API."""

    model_type = "config"

    def __init__(self, **kwargs: Any) -> None:
        self.return_dict = kwargs.pop("return_dict", True)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["model_type"] = self.model_type
        return payload

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        config_path = save_path / "config.json"
        config_path.write_text(self.to_json_string(), encoding="utf-8")
        return config_path

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "PretrainedConfig":
        payload = dict(config_dict)
        payload.pop("model_type", None)
        payload.update(kwargs)
        return cls(**payload)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any) -> "PretrainedConfig":
        config_path = Path(pretrained_model_name_or_path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_dict(config_dict, **kwargs)

