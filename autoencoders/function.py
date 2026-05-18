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


def get_normalization_factory(norm: str, feature_dim: int) -> Callable[[], nn.Module] | None:
    if norm == "none":
        return None
    if norm == "layernorm":
        return lambda: nn.LayerNorm(feature_dim)
    if norm == "batchnorm":
        return lambda: FeatureBatchNorm1d(feature_dim)
    raise ValueError("norm must be one of: 'none', 'layernorm', 'batchnorm'.")


def initialize_linear_weight(linear: nn.Linear, weight_init: str) -> None:
    if weight_init == "default":
        return
    if weight_init == "xavier_uniform":
        nn.init.xavier_uniform_(linear.weight)
    elif weight_init == "xavier_normal":
        nn.init.xavier_normal_(linear.weight)
    else:
        raise ValueError("weight_init must be one of: 'default', 'xavier_uniform', 'xavier_normal'.")
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


@torch.no_grad()
def kmeans_cluster_centers(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError("kmeans_cluster_centers expects a 2D tensor of shape (num_samples, feature_dim).")
    if samples.shape[0] == 0:
        raise ValueError("kmeans_cluster_centers requires at least one sample.")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive.")

    num_samples, feature_dim = samples.shape
    device = samples.device
    dtype = samples.dtype

    if num_samples >= num_clusters:
        initial_indices = torch.randperm(num_samples, device=device)[:num_clusters]
    else:
        initial_indices = torch.randint(0, num_samples, (num_clusters,), device=device)
    centers = samples[initial_indices].clone()

    for _ in range(num_iters):
        distances = (
            samples.pow(2).sum(dim=-1, keepdim=True)
            - 2 * samples @ centers.t()
            + centers.pow(2).sum(dim=-1)
        )
        assignments = distances.argmin(dim=-1)
        updated_centers = centers.clone()

        for cluster_index in range(num_clusters):
            cluster_mask = assignments == cluster_index
            if cluster_mask.any():
                updated_centers[cluster_index] = samples[cluster_mask].mean(dim=0)
            else:
                fallback_index = torch.randint(0, num_samples, (1,), device=device)
                updated_centers[cluster_index] = samples[fallback_index].reshape(feature_dim)

        if torch.allclose(updated_centers, centers):
            centers = updated_centers
            break
        centers = updated_centers

    return centers.to(device=device, dtype=dtype)


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


class FeatureBatchNorm1d(nn.Module):
    """Apply BatchNorm1d to the final feature axis of an arbitrary-rank tensor."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, inputs):  # type: ignore[override]
        original_shape = inputs.shape
        outputs = self.norm(inputs.reshape(-1, self.feature_dim))
        return outputs.reshape(original_shape)
