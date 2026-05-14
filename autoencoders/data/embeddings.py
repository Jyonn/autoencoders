"""Helpers for loading and packaging embedding matrices."""

from __future__ import annotations

import json
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import torch

from .base import AutoencoderDataset


@dataclass
class EmbeddingMatrix:
    """A simple in-memory representation of a token embedding matrix."""

    tokens: list[str]
    matrix: torch.Tensor
    token_to_index: dict[str, int]
    source_path: str | None = None
    name: str | None = None
    texts: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @property
    def num_embeddings(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.matrix.shape[1])


class EmbeddingTensorDataset(AutoencoderDataset):
    """Dataset that exposes each embedding vector as one training sample."""

    def __init__(self, embedding_matrix: EmbeddingMatrix, split: str = "train") -> None:
        super().__init__(split=split)
        self.embedding_matrix = embedding_matrix

    def __len__(self) -> int:
        return self.embedding_matrix.num_embeddings

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.embedding_matrix.matrix[index]


def load_text_embedding_matrix(
    path: str | Path,
    *,
    max_vectors: int | None = None,
    expected_dim: int | None = None,
    skip_first_line: bool = False,
    dtype: torch.dtype = torch.float32,
) -> EmbeddingMatrix:
    """Load a whitespace-separated embedding text file such as GloVe."""

    file_path = Path(path)
    tokens: list[str] = []
    rows: list[list[float]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if skip_first_line and line_number == 1:
                continue
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed embedding row at line {line_number}: {stripped!r}")

            token = parts[0]
            values = parts[1:]

            if expected_dim is None:
                expected_dim = len(values)
            elif len(values) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch at line {line_number}: "
                    f"expected {expected_dim}, got {len(values)}"
                )

            tokens.append(token)
            rows.append([float(value) for value in values])

            if max_vectors is not None and len(tokens) >= max_vectors:
                break

    if not rows:
        raise ValueError(f"No embeddings were loaded from {file_path}.")

    matrix = torch.tensor(rows, dtype=dtype)
    token_to_index = {token: index for index, token in enumerate(tokens)}
    return EmbeddingMatrix(
        tokens=tokens,
        matrix=matrix,
        token_to_index=token_to_index,
        source_path=str(file_path),
        name=file_path.stem,
    )


def save_embedding_artifact(embedding_matrix: EmbeddingMatrix, output_dir: str | Path) -> Path:
    """Save a processed embedding matrix in a compact torch-friendly format."""

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = save_dir / "embeddings.pt"
    tokens_path = save_dir / "tokens.txt"
    texts_path = save_dir / "texts.txt"
    metadata_path = save_dir / "metadata.json"

    torch.save(embedding_matrix.matrix, tensor_path)
    tokens_path.write_text("\n".join(embedding_matrix.tokens) + "\n", encoding="utf-8")
    if embedding_matrix.texts is not None:
        if len(embedding_matrix.texts) != embedding_matrix.num_embeddings:
            raise ValueError("embedding_matrix.texts must align with the number of embeddings.")
        texts_path.write_text("\n".join(embedding_matrix.texts) + "\n", encoding="utf-8")
    metadata = {
        "name": embedding_matrix.name,
        "source_path": embedding_matrix.source_path,
        "num_embeddings": embedding_matrix.num_embeddings,
        "embedding_dim": embedding_matrix.embedding_dim,
    }
    if embedding_matrix.metadata is not None:
        metadata["artifact_metadata"] = embedding_matrix.metadata
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return save_dir


def load_embedding_artifact(path: str | Path) -> EmbeddingMatrix:
    """Load a processed embedding artifact created by save_embedding_artifact()."""

    load_dir = Path(path)
    tensor_path = load_dir / "embeddings.pt"
    tokens_path = load_dir / "tokens.txt"
    texts_path = load_dir / "texts.txt"
    metadata_path = load_dir / "metadata.json"

    matrix = torch.load(tensor_path, map_location="cpu")
    tokens = tokens_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    token_to_index = {token: index for index, token in enumerate(tokens)}
    texts = texts_path.read_text(encoding="utf-8").splitlines() if texts_path.exists() else None

    return EmbeddingMatrix(
        tokens=tokens,
        matrix=matrix,
        token_to_index=token_to_index,
        source_path=metadata.get("source_path"),
        name=metadata.get("name"),
        texts=texts,
        metadata=metadata.get("artifact_metadata"),
    )
