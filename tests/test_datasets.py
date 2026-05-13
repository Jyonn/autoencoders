"""Tests for dataset abstractions and cached GloVe preparation."""

from __future__ import annotations

import io
import tempfile
import unittest
from unittest import mock
import zipfile
from pathlib import Path

import torch

from autoencoders.data import GloVeDataset, create_dataloaders, default_cache_dir, load_dataset, split_dataset


class FakeResponse:
    def __init__(self, payload: bytes, *, content_length: int | None = None, fail_after_reads: int | None = None) -> None:
        self._buffer = io.BytesIO(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self._read_count = 0
        self._fail_after_reads = fail_after_reads

    def read(self, size: int = -1) -> bytes:
        if self._fail_after_reads is not None and self._read_count >= self._fail_after_reads:
            raise RuntimeError("interrupted download")
        self._read_count += 1
        return self._buffer.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._buffer.close()


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

    def test_download_replaces_invalid_cached_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = GloVeDataset(dim=50, max_vectors=2)
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)
                dataset.archive_path.write_bytes(b"not-a-zip")

                valid_zip_bytes = io.BytesIO()
                with zipfile.ZipFile(valid_zip_bytes, "w") as archive:
                    archive.writestr(
                        dataset.vector_filename,
                        "cat " + " ".join(["0.1"] * 50) + "\n"
                        "dog " + " ".join(["0.2"] * 50) + "\n",
                    )
                payload = valid_zip_bytes.getvalue()
                response = FakeResponse(payload, content_length=len(payload))
                progress_stream = io.StringIO()

                with mock.patch("urllib.request.urlopen", return_value=response):
                    with mock.patch("sys.stderr", progress_stream):
                        dataset.download()

                self.assertTrue(dataset.archive_path.exists())
                self.assertFalse(dataset.archive_temp_path.exists())
                self.assertTrue(dataset._is_valid_archive(dataset.archive_path))
                self.assertIn("Downloading glove.6B.zip", progress_stream.getvalue())
                self.assertIn("100%", progress_stream.getvalue())

    def test_prepare_raises_clear_error_for_invalid_archive_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = GloVeDataset(dim=50, max_vectors=2)
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)
                dataset.archive_path.write_bytes(b"not-a-zip")

                with self.assertRaises(zipfile.BadZipFile):
                    dataset.ensure_prepared(download=False)

    def test_download_cleans_up_temp_file_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = GloVeDataset(dim=50, max_vectors=2)
                response = FakeResponse(b"partial-download", content_length=32, fail_after_reads=1)
                progress_stream = io.StringIO()

                with mock.patch("urllib.request.urlopen", return_value=response):
                    with mock.patch("sys.stderr", progress_stream):
                        with self.assertRaises(RuntimeError):
                            dataset.download()

                self.assertFalse(dataset.archive_path.exists())
                self.assertFalse(dataset.archive_temp_path.exists())

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
