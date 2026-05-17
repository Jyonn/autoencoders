"""Shared utility functions used across the autoencoders package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
from torch import nn


def get_activation_factory(activation: str) -> Callable[[], nn.Module]:
    activations: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    return activations[activation]


def set_seed(seed: int) -> None:
    """Set the global torch seed used by training."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_cache_dir() -> Path:
    """Return the default cache directory for downloadable datasets."""

    cache_dir = os.environ.get("AUTOENCODERS_CACHE")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path.home() / ".cache" / "autoencoders"


def format_num_bytes(num_bytes: int) -> str:
    """Format a byte count into a compact human-readable string."""

    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-facing device string into a torch device."""

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)
