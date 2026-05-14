"""Shared utilities for encoder-backed text datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import zipfile
from pathlib import Path
from typing import Iterable

import torch

from .base import CachedDataset, DatasetLoaders, DatasetSplits, create_dataloaders, split_dataset
from .embeddings import (
    EmbeddingMatrix,
    EmbeddingTensorDataset,
    load_embedding_artifact,
    save_embedding_artifact,
)


@dataclass
class TextEmbeddingExample:
    """A single text sample that will be materialized into an embedding."""

    sample_id: str
    text: str


class TextEmbeddingEncoder(ABC):
    """Minimal encoder contract for turning texts into embedding tensors."""

    model_name: str

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode raw texts into a dense float tensor of shape [N, D]."""


class SentenceTransformerTextEncoder(TextEmbeddingEncoder):
    """Sentence-Transformers backend for encoder-backed text datasets."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 128,
        normalize_embeddings: bool = False,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for encoder-backed text datasets. "
                "Install it with `pip install autoencoders[text]` or "
                "`pip install sentence-transformers`."
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.detach().to(dtype=torch.float32, device="cpu")


class EncoderBackedTextDataset(CachedDataset, ABC):
    """Base class for datasets that materialize text embeddings with an encoder."""

    encoder_family = "sentence-transformers"
    default_encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        *,
        encoder_name: str | None = None,
        encoder_batch_size: int = 128,
        normalize_embeddings: bool = False,
        max_vectors: int | None = None,
    ) -> None:
        self.encoder_name = encoder_name or self.default_encoder_name
        self.encoder_batch_size = encoder_batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_vectors = max_vectors
        super().__init__()

    @property
    def artifact_name(self) -> str:
        suffix = "full" if self.max_vectors is None else f"top-{self.max_vectors}"
        encoder_slug = self.encoder_name.replace("/", "--").replace(":", "--")
        return f"{self.dataset_name}-{encoder_slug}-{suffix}"

    @property
    def artifact_dir(self) -> Path:
        return self.processed_dir / self.artifact_name

    def is_prepared(self) -> bool:
        required_files = (
            self.artifact_dir / "embeddings.pt",
            self.artifact_dir / "tokens.txt",
            self.artifact_dir / "texts.txt",
            self.artifact_dir / "metadata.json",
        )
        return all(path.exists() for path in required_files)

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
        return create_dataloaders(
            splits,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def build_encoder(self) -> TextEmbeddingEncoder:
        return SentenceTransformerTextEncoder(
            self.encoder_name,
            batch_size=self.encoder_batch_size,
            normalize_embeddings=self.normalize_embeddings,
        )

    @abstractmethod
    def load_examples(self) -> list[TextEmbeddingExample]:
        """Load raw examples that will be materialized into embeddings."""

    def prepare(self) -> None:
        examples = self.load_examples()
        if self.max_vectors is not None:
            examples = examples[: self.max_vectors]
        if not examples:
            raise ValueError(f"No text examples were loaded for dataset {self.dataset_name!r}.")

        encoder = self.build_encoder()
        texts = [example.text for example in examples]
        matrix = encoder.encode_texts(texts)
        tokens = [example.sample_id for example in examples]
        token_to_index = {token: index for index, token in enumerate(tokens)}

        embedding_matrix = EmbeddingMatrix(
            tokens=tokens,
            texts=texts,
            matrix=matrix,
            token_to_index=token_to_index,
            source_path=str(self.raw_dir),
            name=self.artifact_name,
            metadata={
                "dataset_name": self.dataset_name,
                "encoder_family": self.encoder_family,
                "encoder_name": encoder.model_name,
            },
        )
        save_embedding_artifact(embedding_matrix, self.artifact_dir)


class ZipBackedTextDataset(EncoderBackedTextDataset, ABC):
    """Encoder-backed dataset that downloads one zip archive and parses text files from it."""

    base_url: str
    required_members: tuple[str, ...]

    @property
    def archive_name(self) -> str:
        return Path(self.base_url).name

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / self.archive_name

    def has_raw_data(self) -> bool:
        return self.archive_path.exists()

    def download(self, *, force: bool = False) -> None:
        try:
            self.download_to_cache(
                url=self.base_url,
                destination=self.archive_path,
                validator=self._is_valid_archive,
                description=f"Downloading {self.archive_name}",
                force=force,
            )
        except ValueError as exc:
            raise zipfile.BadZipFile(
                f"Downloaded archive {self.archive_path.with_name(self.archive_path.name + '.tmp')} "
                "is not a valid zip file."
            ) from exc

    def _is_valid_archive(self, path: Path) -> bool:
        if not path.exists() or not zipfile.is_zipfile(path):
            return False

        try:
            with zipfile.ZipFile(path) as zip_handle:
                member_names = set(zip_handle.namelist())
                return all(member in member_names for member in self.required_members) and zip_handle.testzip() is None
        except zipfile.BadZipFile:
            return False

    def read_archive_member_lines(self, member_name: str) -> list[str]:
        if not self._is_valid_archive(self.archive_path):
            raise zipfile.BadZipFile(
                f"Cached archive {self.archive_path} is not a valid zip file. "
                "Delete it and retry, or allow automatic download to recreate it."
            )

        with zipfile.ZipFile(self.archive_path) as zip_handle:
            with zip_handle.open(member_name) as handle:
                return handle.read().decode("utf-8").splitlines()

    @staticmethod
    def deduplicate_texts(texts: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        unique_texts: list[str] = []
        for text in texts:
            normalized = text.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_texts.append(normalized)
        return unique_texts
