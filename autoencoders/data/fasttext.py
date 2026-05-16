"""Dataset helpers for fastText English vectors."""

from __future__ import annotations

import zipfile
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


class FastTextEnglishDatasetConfig(BaseDatasetConfig):
    """Configuration for fastText English vectors."""

    model_type = "fasttext_dataset"


class FastTextEnglishDataset(CachedDataset):
    """Downloadable and cacheable access to fastText English word vectors."""

    dataset_name = "fasttext"
    base_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    config_class = FastTextEnglishDatasetConfig
    config: FastTextEnglishDatasetConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = 300

    @property
    def archive_name(self) -> str:
        return Path(self.base_url).name

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / self.archive_name

    @property
    def vector_filename(self) -> str:
        return "wiki-news-300d-1M.vec"

    @property
    def vector_path(self) -> Path:
        return self.external_dir / self.vector_filename

    @property
    def artifact_name(self) -> str:
        suffix = "full" if self.max_vectors is None else f"top-{self.max_vectors}"
        return f"fasttext-en-{self.dim}d-{suffix}"

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
                raise zipfile.BadZipFile(f"Cached archive {self.archive_path} is not a valid zip file.")
            with zipfile.ZipFile(self.archive_path) as zip_handle:
                zip_handle.extract(self.vector_filename, path=self.external_dir)

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
        if not path.exists() or not zipfile.is_zipfile(path):
            return False
        try:
            with zipfile.ZipFile(path) as zip_handle:
                if self.vector_filename not in zip_handle.namelist():
                    return False
                return zip_handle.testzip() is None
        except zipfile.BadZipFile:
            return False
