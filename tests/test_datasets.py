"""Tests for dataset abstractions and cached GloVe preparation."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
import zipfile
from pathlib import Path

import torch

from autoencoders.data import GloVeDataset, create_dataloaders, default_cache_dir, load_dataset, split_dataset


class DatasetUtilitiesTest(unittest.TestCase):
    def test_default_cache_dir_uses_environment_override(self) -> None:
        with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": "/tmp/autoencoders-cache"}, clear=False):
            self.assertEqual(default_cache_dir(), Path("/tmp/autoencoders-cache"))

    def test_glove_dataset_prepare_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = GloVeDataset(dim=50, max_vectors=2)
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(dataset.archive_path, "w") as archive:
                    archive.writestr(
                        dataset.vector_filename,
                        "cat " + " ".join(["0.1"] * 50) + "\n"
                        "dog " + " ".join(["0.2"] * 50) + "\n"
                        "car " + " ".join(["0.3"] * 50) + "\n",
                    )

                artifact_dir = dataset.ensure_prepared(download=False)
                self.assertTrue(artifact_dir.exists())

                embedding_matrix = dataset.load_embedding_matrix(download=False)
                self.assertEqual(embedding_matrix.num_embeddings, 2)
                self.assertEqual(embedding_matrix.embedding_dim, 50)

                artifact_mtime = (artifact_dir / "embeddings.pt").stat().st_mtime
                dataset.ensure_prepared(download=False)
                self.assertEqual((artifact_dir / "embeddings.pt").stat().st_mtime, artifact_mtime)

    def test_load_dataset_returns_glove_dataset(self) -> None:
        dataset = load_dataset("glove", dim=50, max_vectors=10)
        self.assertIsInstance(dataset, GloVeDataset)
        self.assertEqual(dataset.dim, 50)
        self.assertEqual(dataset.max_vectors, 10)

    def test_split_and_dataloaders(self) -> None:
        tensor_dataset = torch.utils.data.TensorDataset(torch.randn(20, 4))

        class FlattenedTensorDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(tensor_dataset)

            def __getitem__(self, index: int) -> torch.Tensor:
                return tensor_dataset[index][0]

        dataset = FlattenedTensorDataset()
        splits = split_dataset(dataset, validation_ratio=0.2, test_ratio=0.1, seed=7)
        loaders = create_dataloaders(splits, batch_size=4)

        self.assertEqual(len(splits.train), 14)
        self.assertEqual(len(splits.validation), 4)
        self.assertEqual(len(splits.test), 2)
        self.assertEqual(tuple(next(iter(loaders.train)).shape)[1], 4)


if __name__ == "__main__":
    unittest.main()
