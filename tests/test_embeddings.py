"""Tests for embedding matrix utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from autoencoders.data import (
    EmbeddingMatrix,
    EmbeddingTensorDataset,
    load_embedding_artifact,
    load_text_embedding_matrix,
    save_embedding_artifact,
)


class EmbeddingMatrixTest(unittest.TestCase):
    def test_load_text_embedding_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "tiny_glove.txt"
            txt_path.write_text("cat 0.1 0.2 0.3\nDog -1.0 0.0 1.5\n", encoding="utf-8")

            embedding_matrix = load_text_embedding_matrix(txt_path, expected_dim=3)

        self.assertEqual(embedding_matrix.tokens, ["cat", "Dog"])
        self.assertEqual(tuple(embedding_matrix.matrix.shape), (2, 3))
        self.assertEqual(embedding_matrix.token_to_index["Dog"], 1)

    def test_save_and_load_embedding_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "tiny_glove.txt"
            txt_path.write_text("cat 0.1 0.2 0.3\nDog -1.0 0.0 1.5\n", encoding="utf-8")

            embedding_matrix = load_text_embedding_matrix(txt_path, expected_dim=3)
            artifact_dir = Path(tmpdir) / "artifact"
            save_embedding_artifact(embedding_matrix, artifact_dir)
            loaded = load_embedding_artifact(artifact_dir)

        self.assertEqual(loaded.tokens, ["cat", "Dog"])
        self.assertTrue(torch.equal(loaded.matrix, embedding_matrix.matrix))
        self.assertEqual(loaded.embedding_dim, 3)

    def test_save_and_load_embedding_artifact_with_texts_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_matrix = EmbeddingMatrix(
                tokens=["sample-0", "sample-1"],
                texts=["hello world", "general kenobi"],
                matrix=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
                token_to_index={"sample-0": 0, "sample-1": 1},
                name="toy-sentences",
                metadata={"encoder_name": "sentence-transformers/all-MiniLM-L6-v2"},
            )
            artifact_dir = Path(tmpdir) / "artifact"
            save_embedding_artifact(embedding_matrix, artifact_dir)
            loaded = load_embedding_artifact(artifact_dir)

        self.assertEqual(loaded.tokens, ["sample-0", "sample-1"])
        self.assertEqual(loaded.texts, ["hello world", "general kenobi"])
        self.assertEqual(loaded.metadata, {"encoder_name": "sentence-transformers/all-MiniLM-L6-v2"})
        self.assertTrue(torch.equal(loaded.matrix, embedding_matrix.matrix))

    def test_load_text_embedding_matrix_can_skip_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "tiny_fasttext.vec"
            txt_path.write_text("2 3\ncat 0.1 0.2 0.3\nDog -1.0 0.0 1.5\n", encoding="utf-8")

            embedding_matrix = load_text_embedding_matrix(txt_path, expected_dim=3, skip_first_line=True)

        self.assertEqual(embedding_matrix.tokens, ["cat", "Dog"])
        self.assertEqual(tuple(embedding_matrix.matrix.shape), (2, 3))

    def test_embedding_tensor_dataset_returns_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "tiny_glove.txt"
            txt_path.write_text("cat 0.1 0.2 0.3\nDog -1.0 0.0 1.5\n", encoding="utf-8")
            embedding_matrix = load_text_embedding_matrix(txt_path, expected_dim=3)

        dataset = EmbeddingTensorDataset(embedding_matrix)

        self.assertEqual(len(dataset), 2)
        self.assertTrue(torch.equal(dataset[1], torch.tensor([-1.0, 0.0, 1.5])))


if __name__ == "__main__":
    unittest.main()
