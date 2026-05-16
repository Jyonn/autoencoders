"""Dataset helpers for ConceptNet Numberbatch embeddings."""

from __future__ import annotations

import gzip
from pathlib import Path

from .base import (
    BaseDatasetConfig,
    CachedDataset,
    DatasetLoaders,
    DatasetSplits,
    create_dataloaders,
    split_dataset,
)
from .embeddings import (
    EmbeddingMatrix,
    EmbeddingTensorDataset,
    load_embedding_artifact,
    load_text_embedding_matrix,
    save_embedding_artifact,
)


class ConceptNetNumberbatchDatasetConfig(BaseDatasetConfig):
    """Configuration for ConceptNet Numberbatch embeddings."""

    model_type = "numberbatch_dataset"

    def __init__(
        self,
        *,
        dim: int = 300,
        root: str | Path | None = None,
        max_vectors: int | None = None,
        **kwargs,
    ) -> None:
        self.dim = dim
        super().__init__(root=root, max_vectors=max_vectors, **kwargs)


class ConceptNetNumberbatchDataset(CachedDataset):
    """Downloadable and cacheable access to ConceptNet Numberbatch vectors."""

    dataset_name = "numberbatch"
    base_url = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
    config_class = ConceptNetNumberbatchDatasetConfig

    def __init__(
        self,
        config: ConceptNetNumberbatchDatasetConfig | None = None,
        **kwargs,
    ) -> None:
        config = self.config_class(**kwargs) if config is None else config
        self.config = config
        dim = config.dim
        if dim != 300:
            raise ValueError("dim must be 300 for ConceptNet Numberbatch.")
        self.dim = dim
        self.max_vectors = config.max_vectors
        super().__init__(root=config.root)

    @property
    def archive_name(self) -> str:
        return Path(self.base_url).name

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / self.archive_name

    @property
    def vector_filename(self) -> str:
        return "numberbatch-en-19.08.txt"

    @property
    def vector_path(self) -> Path:
        return self.external_dir / self.vector_filename

    @property
    def artifact_name(self) -> str:
        suffix = "full" if self.max_vectors is None else f"top-{self.max_vectors}"
        return f"numberbatch-en-{self.dim}d-{suffix}"

    @property
    def artifact_dir(self) -> Path:
        return self.processed_dir / self.artifact_name

    def is_prepared(self) -> bool:
        required_files = (
            self.artifact_dir / "embeddings.pt",
            self.artifact_dir / "tokens.txt",
            self.artifact_dir / "metadata.json",
        )
        return all(path.exists() for path in required_files)

    def has_raw_data(self) -> bool:
        return self.archive_path.exists() or self.vector_path.exists()

    def download(self, *, force: bool = False) -> None:
        self.download_to_cache(
            url=self.base_url,
            destination=self.archive_path,
            validator=self._is_valid_archive,
            description=f"Downloading {self.archive_name}",
            force=force,
        )

    def prepare(self) -> None:
        self.external_dir.mkdir(parents=True, exist_ok=True)
        if not self.vector_path.exists():
            if not self._is_valid_archive(self.archive_path):
                raise ValueError(f"Cached archive {self.archive_path} is not a valid gzip file.")
            with gzip.open(self.archive_path, "rb") as compressed, self.vector_path.open("wb") as output:
                output.write(compressed.read())

        embedding_matrix = load_text_embedding_matrix(
            self.vector_path,
            max_vectors=self.max_vectors,
            expected_dim=self.dim,
            skip_first_line=True,
        )
        embedding_matrix.name = self.artifact_name
        save_embedding_artifact(embedding_matrix, self.artifact_dir)

    def load_embedding_matrix(self, *, download: bool = True) -> EmbeddingMatrix:
        self.ensure_prepared(download=download)
        return load_embedding_artifact(self.artifact_dir)

    def as_dataset(self, *, download: bool = True) -> EmbeddingTensorDataset:
        return EmbeddingTensorDataset(self.load_embedding_matrix(download=download))

    def get_splits(
        self,
        *,
        download: bool = True,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplits:
        dataset = self.as_dataset(download=download)
        return split_dataset(dataset, validation_ratio=validation_ratio, test_ratio=test_ratio, seed=seed)

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

    def _is_valid_archive(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with gzip.open(path, "rb") as handle:
                handle.read(8)
            return True
        except OSError:
            return False
