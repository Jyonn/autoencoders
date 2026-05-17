"""Downloadable CIFAR-10 image dataset for CNN-backed autoencoders."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import tarfile

import torch

from .base import (
    AutoencoderDataset,
    BaseDatasetConfig,
    CachedDataset,
    DatasetLoaders,
    DatasetSplits,
    TensorSpec,
    create_dataloaders,
    split_dataset,
)


class CIFAR10DatasetConfig(BaseDatasetConfig):
    """Configuration for the CIFAR-10 image dataset."""

    model_type = "cifar10_dataset"

    def __init__(
        self,
        *,
        max_examples: int | None = None,
        **kwargs,
    ) -> None:
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive when provided.")
        self.max_examples = max_examples
        super().__init__(**kwargs)


class CIFAR10TensorDataset(AutoencoderDataset):
    """Dataset that exposes one CIFAR-10 image tensor per sample."""

    def __init__(self, images: torch.Tensor, split: str = "train") -> None:
        super().__init__(split=split)
        self.images = images

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]

    def get_sample_spec(self) -> TensorSpec:
        return TensorSpec(shape=tuple(int(dimension) for dimension in self.images.shape[1:]))


_CIFAR10_ARCHIVE_MEMBER_PREFIX = "cifar-10-batches-py"
_CIFAR10_BATCH_MEMBERS = tuple(
    f"{_CIFAR10_ARCHIVE_MEMBER_PREFIX}/data_batch_{index}" for index in range(1, 6)
) + (f"{_CIFAR10_ARCHIVE_MEMBER_PREFIX}/test_batch",)


class CIFAR10Dataset(CachedDataset):
    """Download and cache CIFAR-10 as normalized HWC tensors."""

    dataset_name = "cifar10"
    base_urls = (
        "https://mirrors.bfsu.edu.cn/osdn/datasets/74526/cifar-10-python.tar.gz",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    )
    archive_member_prefix = _CIFAR10_ARCHIVE_MEMBER_PREFIX
    batch_members = _CIFAR10_BATCH_MEMBERS
    config_class = CIFAR10DatasetConfig
    config: CIFAR10DatasetConfig

    @property
    def archive_name(self) -> str:
        return Path(self.base_urls[0]).name

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / self.archive_name

    @property
    def artifact_name(self) -> str:
        suffix = "full" if self.config.max_examples is None else f"top-{self.config.max_examples}"
        return f"cifar10-{suffix}"

    @property
    def artifact_dir(self) -> Path:
        return self.processed_dir / self.artifact_name

    def is_prepared(self) -> bool:
        required_files = (
            self.artifact_dir / "images.pt",
            self.artifact_dir / "labels.pt",
            self.artifact_dir / "metadata.json",
        )
        return all(path.exists() for path in required_files)

    def has_raw_data(self) -> bool:
        return self.archive_path.exists()

    def download(self, *, force: bool = False) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            for url in self.base_urls:
                try:
                    self.download_to_cache(
                        url=url,
                        destination=self.archive_path,
                        validator=self._is_valid_archive,
                        description=f"Downloading {self.archive_name} (attempt {attempt}, {Path(url).parent.name})",
                        force=force or attempt > 1,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if self.archive_path.exists():
                        self.archive_path.unlink()
        raise RuntimeError(
            f"Failed to download CIFAR-10 after trying {len(self.base_urls)} mirrors for 3 attempts."
        ) from last_error

    def prepare(self) -> None:
        images, labels = self._load_images_and_labels_from_archive()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(images, self.artifact_dir / "images.pt")
        torch.save(labels, self.artifact_dir / "labels.pt")
        metadata = {
            "num_examples": int(images.shape[0]),
            "shape": list(images.shape[1:]),
            "dataset_name": self.dataset_name,
        }
        (self.artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def get_sample_spec(self) -> TensorSpec:
        return TensorSpec(shape=(32, 32, 3))

    def as_dataset(self, *, download: bool = True) -> CIFAR10TensorDataset:
        images, _ = self.load_tensors(download=download)
        return CIFAR10TensorDataset(images)

    def get_splits(
        self,
        *,
        download: bool = True,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplits:
        dataset = self.as_dataset(download=download)
        return split_dataset(
            dataset,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    def get_dataloaders(
        self,
        *,
        download: bool = True,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> DatasetLoaders:
        splits = self.get_splits(
            download=download,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        return create_dataloaders(splits, batch_size=batch_size, num_workers=num_workers)

    def load_tensors(self, *, download: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        self.ensure_prepared(download=download)
        images = torch.load(self.artifact_dir / "images.pt", map_location="cpu")
        labels = torch.load(self.artifact_dir / "labels.pt", map_location="cpu")
        return images, labels

    def _load_images_and_labels_from_archive(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._is_valid_archive(self.archive_path):
            raise tarfile.ReadError(
                f"Cached archive {self.archive_path} is not a valid CIFAR-10 tarball."
            )

        all_images: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        with tarfile.open(self.archive_path, "r:gz") as archive:
            member_names = {member.name for member in archive.getmembers()}
            present_members = [member_name for member_name in self.batch_members if member_name in member_names]
            if not present_members:
                raise ValueError(f"Archive {self.archive_path} does not contain CIFAR-10 batch files.")

            for member_name in present_members:
                member = archive.getmember(member_name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Failed to extract {member_name} from {self.archive_path}.")
                batch = pickle.load(extracted, encoding="bytes")
                raw_images = torch.tensor(batch[b"data"], dtype=torch.float32).reshape(-1, 3, 32, 32)
                images = raw_images.permute(0, 2, 3, 1).contiguous() / 255.0
                labels = torch.tensor(batch[b"labels"], dtype=torch.long)
                all_images.append(images)
                all_labels.append(labels)

        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0)
        if self.config.max_examples is not None:
            images = images[: self.config.max_examples]
            labels = labels[: self.config.max_examples]
        return images, labels

    def _is_valid_archive(self, path: Path) -> bool:
        if not path.exists() or not tarfile.is_tarfile(path):
            return False
        try:
            with tarfile.open(path, "r:gz") as archive:
                member_names = {member.name for member in archive.getmembers()}
            return any(member_name in member_names for member_name in self.batch_members)
        except (tarfile.TarError, EOFError, OSError):
            return False
