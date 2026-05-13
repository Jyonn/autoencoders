"""Dataset loading helpers."""

from __future__ import annotations

from typing import Any

from .fasttext import FastTextEnglishDataset
from .glove import GloVeDataset
from .numberbatch import ConceptNetNumberbatchDataset


def load_dataset(name: str, **kwargs: Any) -> GloVeDataset | FastTextEnglishDataset | ConceptNetNumberbatchDataset:
    """Load a named dataset, downloading and caching it on demand."""

    if name == "glove":
        return GloVeDataset(**kwargs)
    if name == "fasttext":
        return FastTextEnglishDataset(**kwargs)
    if name == "numberbatch":
        return ConceptNetNumberbatchDataset(**kwargs)
    raise ValueError(f"Unknown dataset {name!r}. Available datasets: 'glove', 'fasttext', 'numberbatch'.")
