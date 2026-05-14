"""PyTorch model utilities for the autoencoders library."""

from __future__ import annotations

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
    def from_config(cls, config: Any) -> "PreTrainedAutoencoderModel":
        return cls(config)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_path)
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
        config = cls.config_class.from_pretrained(load_path, **kwargs)
        model = cls.from_config(config)

        weights_path = load_path / cls.weights_name if load_path.is_dir() else load_path
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=map_location)
            model.load_state_dict(state_dict)

        return model
