"""Base dataset abstractions for autoencoder training and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..configuration_utils import PretrainedConfig


class DataSpec(ABC):
    """Base structural description for dataset samples and module I/O."""

    @abstractmethod
    def matches(self, other: "DataSpec") -> bool:
        """Return whether this spec is structurally compatible with another spec."""


@dataclass(frozen=True)
class TensorSpec(DataSpec):
    """Description of a tensor-valued sample without a batch dimension."""

    shape: tuple[int | None, ...]

    def __post_init__(self) -> None:
        normalized_shape = tuple(self.shape)
        for index, dim in enumerate(normalized_shape):
            if dim is not None and dim <= 0:
                raise ValueError(f"TensorSpec.shape[{index}] must be positive or None, got {dim!r}.")
        object.__setattr__(self, "shape", normalized_shape)

    def matches(self, other: DataSpec) -> bool:
        if not isinstance(other, TensorSpec):
            return False
        if len(self.shape) != len(other.shape):
            return False
        for left_dim, right_dim in zip(self.shape, other.shape):
            if left_dim is not None and right_dim is not None and left_dim != right_dim:
                return False
        return True


@dataclass(frozen=True)
class ListSpec(DataSpec):
    """Description of a repeated homogeneous structure."""

    element_spec: DataSpec
    num_elements: int | None = None

    def __post_init__(self) -> None:
        if self.num_elements is not None and self.num_elements <= 0:
            raise ValueError("ListSpec.num_elements must be positive or None.")

    def matches(self, other: DataSpec) -> bool:
        if not isinstance(other, ListSpec):
            return False
        if (
            self.num_elements is not None
            and other.num_elements is not None
            and self.num_elements != other.num_elements
        ):
            return False
        return self.element_spec.matches(other.element_spec)


@dataclass(frozen=True)
class DictSpec(DataSpec):
    """Description of a structured dictionary sample."""

    elements: dict[str, DataSpec]
    restrict_keys: bool = True

    def matches(self, other: DataSpec) -> bool:
        if not isinstance(other, DictSpec):
            return False
        if self.restrict_keys and set(self.elements) != set(other.elements):
            return False
        for key, spec in self.elements.items():
            other_spec = other.elements.get(key)
            if other_spec is None or not spec.matches(other_spec):
                return False
        return True


class BaseDatasetConfig(PretrainedConfig):
    """Base configuration shared by built-in datasets."""

    model_type = "dataset"

    def __init__(
        self,
        *,
        max_vectors: int | None = None,
        **kwargs,
    ) -> None:
        self.max_vectors = max_vectors
        super().__init__(**kwargs)


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


class DownloadProgressBar:
    """A small terminal progress bar for dataset downloads."""

    def __init__(self, description: str, total_bytes: int | None, stream=None) -> None:
        self.description = description
        self.total_bytes = total_bytes
        self.stream = sys.stderr if stream is None else stream
        self.downloaded_bytes = 0
        self._finished = False
        self._render()

    def update(self, chunk_size: int) -> None:
        self.downloaded_bytes += chunk_size
        self._render()

    def close(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._render(final=True)

    def _render(self, final: bool = False) -> None:
        if self.total_bytes is not None and self.total_bytes > 0:
            ratio = min(self.downloaded_bytes / self.total_bytes, 1.0)
            filled = int(ratio * 20)
            bar = "=" * filled + "." * (20 - filled)
            percent = int(ratio * 100)
            message = (
                f"\r{self.description} [{bar}] {percent:3d}% "
                f"{format_num_bytes(self.downloaded_bytes)}/{format_num_bytes(self.total_bytes)}"
            )
        else:
            message = f"\r{self.description} {format_num_bytes(self.downloaded_bytes)}"

        if final:
            message += "\n"

        self.stream.write(message)
        self.stream.flush()


class ItemProgressBar:
    """A small terminal progress bar for item-based preprocessing work."""

    def __init__(self, description: str, total_items: int, stream=None) -> None:
        self.description = description
        self.total_items = max(total_items, 0)
        self.stream = sys.stderr if stream is None else stream
        self.completed_items = 0
        self._finished = False
        self._render()

    def update(self, item_count: int) -> None:
        self.completed_items = min(self.completed_items + item_count, self.total_items)
        self._render()

    def close(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._render(final=True)

    def _render(self, final: bool = False) -> None:
        if self.total_items > 0:
            ratio = min(self.completed_items / self.total_items, 1.0)
            filled = int(ratio * 20)
            bar = "=" * filled + "." * (20 - filled)
            percent = int(ratio * 100)
            message = (
                f"\r{self.description} [{bar}] {percent:3d}% "
                f"{self.completed_items}/{self.total_items}"
            )
        else:
            message = f"\r{self.description} 0/0"

        if final:
            message += "\n"

        self.stream.write(message)
        self.stream.flush()


class AutoencoderDataset(Dataset[object]):
    """Simple dataset contract for autoencoder-friendly tensor samples."""

    split: str

    def __init__(self, split: str = "train") -> None:
        self.split = split

    @abstractmethod
    def get_sample_spec(self) -> DataSpec:
        """Return the structural description for one dataset sample."""


class CachedDataset(ABC):
    """Base class for datasets that download raw files and cache processed artifacts."""

    dataset_name = "dataset"
    config_class = BaseDatasetConfig
    config: BaseDatasetConfig

    def __init__(self, **kwargs) -> None:
        config = kwargs.pop("config")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"{self.__class__.__name__} received unexpected keyword arguments: {unknown}")
        self.config = config
        self.root = default_cache_dir()
        self.dataset_dir = self.root / self.dataset_name
        self.raw_dir = self.dataset_dir / "raw"
        self.external_dir = self.dataset_dir / "external"
        self.processed_dir = self.dataset_dir / "processed"

    @property
    def max_vectors(self) -> int | None:
        return getattr(self.config, "max_vectors", None)

    @abstractmethod
    def get_sample_spec(self, *, download: bool = True) -> DataSpec:
        """Return the structural description of one prepared sample."""

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

    def download_to_cache(
        self,
        *,
        url: str,
        destination: Path,
        validator: Callable[[Path], bool] | None = None,
        description: str | None = None,
        force: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        """Download a file atomically with progress reporting and cache validation."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f"{destination.name}.tmp")
        self._cleanup_temp_file(temp_path)

        if destination.exists() and not force:
            if validator is None or validator(destination):
                return destination
            destination.unlink()
        elif destination.exists():
            destination.unlink()

        try:
            with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
                total_bytes = self._response_content_length(response)
                progress = DownloadProgressBar(description or destination.name, total_bytes)
                try:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        progress.update(len(chunk))
                finally:
                    progress.close()
        except Exception:
            self._cleanup_temp_file(temp_path)
            raise

        if validator is not None and not validator(temp_path):
            self._cleanup_temp_file(temp_path)
            raise ValueError(f"Downloaded file {temp_path} failed validation.")

        temp_path.replace(destination)
        return destination

    @staticmethod
    def _cleanup_temp_file(path: Path) -> None:
        if path.exists():
            path.unlink()

    @staticmethod
    def _response_content_length(response) -> int | None:
        length = response.headers.get("Content-Length")
        if length is None:
            return None
        try:
            return int(length)
        except (TypeError, ValueError):
            return None

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

    train: Dataset[object]
    validation: Dataset[object]
    test: Dataset[object]


def split_dataset(
    dataset: Dataset[object],
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

    train: DataLoader[object]
    validation: DataLoader[object]
    test: DataLoader[object]


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
