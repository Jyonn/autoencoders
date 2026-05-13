"""Dataset loading helpers."""

from __future__ import annotations

from typing import Any

from .glove import GloVeDataset


def load_dataset(name: str, **kwargs: Any) -> GloVeDataset:
    """Load a named dataset, downloading and caching it on demand."""

    normalized_name = name.strip().lower()
    if normalized_name == "glove":
        return GloVeDataset(**kwargs)
    raise ValueError(f"Unknown dataset {name!r}. Available datasets: 'glove'.")
