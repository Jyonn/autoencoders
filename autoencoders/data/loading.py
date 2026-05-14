"""Dataset loading helpers."""

from __future__ import annotations

from typing import Any

from .base import CachedDataset
from .fasttext import FastTextEnglishDataset
from .glove import GloVeDataset
from .multinli import MultiNLIDataset
from .numberbatch import ConceptNetNumberbatchDataset
from .snli import SNLIDataset


def load_dataset(name: str, **kwargs: Any) -> CachedDataset:
    """Load a named dataset, downloading and caching it on demand."""

    if name == "glove":
        return GloVeDataset(**kwargs)
    if name == "fasttext":
        return FastTextEnglishDataset(**kwargs)
    if name == "numberbatch":
        return ConceptNetNumberbatchDataset(**kwargs)
    if name == "snli":
        return SNLIDataset(**kwargs)
    if name == "multinli":
        return MultiNLIDataset(**kwargs)
    raise ValueError(
        "Unknown dataset "
        f"{name!r}. Available datasets: 'glove', 'fasttext', 'numberbatch', 'snli', 'multinli'."
    )
