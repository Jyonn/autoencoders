"""Base dataset abstractions for autoencoder training and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def default_cache_dir() -> Path:
    """Return the default cache directory for downloadable datasets."""

    cache_dir = os.environ.get("AUTOENCODERS_CACHE")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path.home() / ".cache" / "autoencoders"


class AutoencoderDataset(Dataset[torch.Tensor]):
    """Simple dataset contract for autoencoder-friendly tensor samples."""

    split: str

    def __init__(self, split: str = "train") -> None:
        self.split = split


class CachedDataset(ABC):
    """Base class for datasets that download raw files and cache processed artifacts."""

    dataset_name = "dataset"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()
        self.dataset_dir = self.root / self.dataset_name
        self.raw_dir = self.dataset_dir / "raw"
        self.external_dir = self.dataset_dir / "external"
        self.processed_dir = self.dataset_dir / "processed"

    def ensure_prepared(
        self,
        *,
        download: bool = True,
        force_download: bool = False,
        force_prepare: bool = False,
    ) -> Path:
        """Ensure the processed artifact exists and return its directory."""

        artifact_dir = self.artifact_dir
        if self.is_prepared() and not force_prepare:
            return artifact_dir

        if download:
            self.download(force=force_download)
        elif not self.has_raw_data():
            raise FileNotFoundError(
                f"Raw data for {self.dataset_name!r} is missing under {self.raw_dir}."
            )

        self.prepare()
        return artifact_dir

    def is_prepared(self) -> bool:
        """Return True when the processed artifact is complete and ready to use."""

        return self.artifact_dir.exists()

    @property
    @abstractmethod
    def artifact_dir(self) -> Path:
        """Directory containing the processed dataset artifact."""

    @abstractmethod
    def has_raw_data(self) -> bool:
        """Return True when enough raw files are present to prepare the dataset."""

    @abstractmethod
    def download(self, *, force: bool = False) -> None:
        """Download the raw dataset files into the cache."""

    @abstractmethod
    def prepare(self) -> None:
        """Convert raw files into a processed artifact."""


@dataclass
class DatasetSplits:
    """A deterministic set of dataset splits."""

    train: Dataset[torch.Tensor]
    validation: Dataset[torch.Tensor]
    test: Dataset[torch.Tensor]


def split_dataset(
    dataset: Dataset[torch.Tensor],
    *,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplits:
    """Split a dataset into train, validation, and test subsets."""

    if validation_ratio < 0 or test_ratio < 0:
        raise ValueError("validation_ratio and test_ratio must be non-negative.")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1.0.")

    num_examples = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_examples, generator=generator).tolist()

    num_validation = int(num_examples * validation_ratio)
    num_test = int(num_examples * test_ratio)
    num_train = num_examples - num_validation - num_test

    train_indices = permutation[:num_train]
    validation_indices = permutation[num_train : num_train + num_validation]
    test_indices = permutation[num_train + num_validation :]

    return DatasetSplits(
        train=Subset(dataset, train_indices),
        validation=Subset(dataset, validation_indices),
        test=Subset(dataset, test_indices),
    )


@dataclass
class DatasetLoaders:
    """Convenience wrapper for train, validation, and test dataloaders."""

    train: DataLoader[torch.Tensor]
    validation: DataLoader[torch.Tensor]
    test: DataLoader[torch.Tensor]


def create_dataloaders(
    splits: DatasetSplits,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DatasetLoaders:
    """Create dataloaders from deterministic dataset splits."""

    return DatasetLoaders(
        train=DataLoader(splits.train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        validation=DataLoader(
            splits.validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
