"""Tests for dataset abstractions and cached GloVe preparation."""

from __future__ import annotations

import io
import tempfile
import unittest
from unittest import mock
import zipfile
import gzip
import json
from pathlib import Path

import torch

from autoencoders.data import (
    ConceptNetNumberbatchDatasetConfig,
    ConceptNetNumberbatchDataset,
    FastTextEnglishDatasetConfig,
    FastTextEnglishDataset,
    Flickr30kDatasetConfig,
    Flickr30kDataset,
    GloVeDatasetConfig,
    GloVeDataset,
    MultiNLIDatasetConfig,
    MultiNLIDataset,
    SNLIDatasetConfig,
    SNLIDataset,
    create_dataloaders,
    default_cache_dir,
    load_dataset,
    split_dataset,
)
from autoencoders.data.base import ItemProgressBar


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


class FakeTextEncoder:
    def __init__(self, model_name: str = "fake-encoder") -> None:
        self.model_name = model_name

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for index, text in enumerate(texts):
            rows.append([float(len(text)), float(index), float(len(text.split())), 1.0])
        return torch.tensor(rows, dtype=torch.float32)


class FakeCLIPEncoder:
    def __init__(self, model_name: str = "fake-clip", pretrained_name: str = "fake-pretrained") -> None:
        self.model_name = model_name
        self.pretrained_name = pretrained_name

    def encode_images(self, image_paths: list[Path]) -> torch.Tensor:
        rows = []
        for index, path in enumerate(image_paths):
            rows.append([float(index + 1), float(len(path.name)), 0.0, 1.0])
        return torch.tensor(rows, dtype=torch.float32)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for index, text in enumerate(texts):
            rows.append([float(len(text)), float(index + 1), 1.0, 0.0])
        return torch.tensor(rows, dtype=torch.float32)


class DatasetUtilitiesTest(unittest.TestCase):
    def test_item_progress_bar_reports_completion(self) -> None:
        stream = io.StringIO()
        progress = ItemProgressBar("Encoding captions", 5, stream=stream)
        progress.update(2)
        progress.update(3)
        progress.close()
        rendered = stream.getvalue()
        self.assertIn("Encoding captions", rendered)
        self.assertIn("100%", rendered)
        self.assertIn("5/5", rendered)

    def test_item_progress_bar_does_not_fake_completion_on_early_close(self) -> None:
        stream = io.StringIO()
        progress = ItemProgressBar("Encoding captions", 5, stream=stream)
        progress.update(2)
        progress.close()
        rendered = stream.getvalue().splitlines()[-1]
        self.assertIn("2/5", rendered)

    def test_default_cache_dir_uses_environment_override(self) -> None:
        with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": "/tmp/autoencoders-cache"}, clear=False):
            self.assertEqual(default_cache_dir(), Path("/tmp/autoencoders-cache"))

    def test_glove_dataset_prepare_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = GloVeDataset(config=GloVeDatasetConfig(dim=50, max_vectors=2))
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
                dataset = GloVeDataset(config=GloVeDatasetConfig(dim=50, max_vectors=2))
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
                dataset = GloVeDataset(config=GloVeDatasetConfig(dim=50, max_vectors=2))
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)
                dataset.archive_path.write_bytes(b"not-a-zip")

                with self.assertRaises(zipfile.BadZipFile):
                    dataset.ensure_prepared(download=False)

    def test_download_cleans_up_temp_file_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = GloVeDataset(config=GloVeDatasetConfig(dim=50, max_vectors=2))
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

    def test_fasttext_dataset_prepare_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = FastTextEnglishDataset(config=FastTextEnglishDatasetConfig(max_vectors=2))
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(dataset.archive_path, "w") as archive:
                    archive.writestr(
                        dataset.vector_filename,
                        "3 300\n"
                        "cat " + " ".join(["0.1"] * 300) + "\n"
                        "dog " + " ".join(["0.2"] * 300) + "\n"
                        "car " + " ".join(["0.3"] * 300) + "\n",
                    )

                artifact_dir = dataset.ensure_prepared(download=False)
                self.assertTrue(artifact_dir.exists())
                embedding_matrix = dataset.load_embedding_matrix(download=False)
                self.assertEqual(embedding_matrix.num_embeddings, 2)
                self.assertEqual(embedding_matrix.embedding_dim, 300)

    def test_numberbatch_dataset_prepare_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = ConceptNetNumberbatchDataset(config=ConceptNetNumberbatchDatasetConfig(max_vectors=2))
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)

                payload = (
                    "3 300\n"
                    "/c/en/cat " + " ".join(["0.1"] * 300) + "\n"
                    "/c/en/dog " + " ".join(["0.2"] * 300) + "\n"
                    "/c/en/car " + " ".join(["0.3"] * 300) + "\n"
                ).encode("utf-8")
                with gzip.open(dataset.archive_path, "wb") as archive:
                    archive.write(payload)

                artifact_dir = dataset.ensure_prepared(download=False)
                self.assertTrue(artifact_dir.exists())
                embedding_matrix = dataset.load_embedding_matrix(download=False)
                self.assertEqual(embedding_matrix.num_embeddings, 2)
                self.assertEqual(embedding_matrix.embedding_dim, 300)

    def test_load_dataset_returns_fasttext_and_numberbatch(self) -> None:
        fasttext = load_dataset("fasttext", max_vectors=10)
        numberbatch = load_dataset("numberbatch", max_vectors=10)
        self.assertIsInstance(fasttext, FastTextEnglishDataset)
        self.assertIsInstance(numberbatch, ConceptNetNumberbatchDataset)

    def test_snli_dataset_prepare_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = SNLIDataset(config=SNLIDatasetConfig(max_vectors=3))
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(dataset.archive_path, "w") as archive:
                    archive.writestr(
                        "snli_1.0/snli_1.0_train.jsonl",
                        "\n".join(
                            [
                                json.dumps({"sentence1": "A cat sits.", "sentence2": "An animal rests."}),
                                json.dumps({"sentence1": "A cat sits.", "sentence2": "A dog runs."}),
                            ]
                        )
                        + "\n",
                    )
                    archive.writestr(
                        "snli_1.0/snli_1.0_dev.jsonl",
                        json.dumps({"sentence1": "Birds fly.", "sentence2": "Creatures move."}) + "\n",
                    )
                    archive.writestr(
                        "snli_1.0/snli_1.0_test.jsonl",
                        json.dumps({"sentence1": "Snow falls.", "sentence2": "Weather changes."}) + "\n",
                    )

                with mock.patch.object(dataset, "build_encoder", return_value=FakeTextEncoder("fake-snli")):
                    artifact_dir = dataset.ensure_prepared(download=False)
                    self.assertTrue(artifact_dir.exists())
                    embedding_matrix = dataset.load_embedding_matrix(download=False)

                self.assertEqual(embedding_matrix.num_embeddings, 3)
                self.assertEqual(embedding_matrix.embedding_dim, 4)
                self.assertEqual(embedding_matrix.metadata["encoder_name"], "fake-snli")
                self.assertEqual(embedding_matrix.texts, ["A cat sits.", "An animal rests.", "A dog runs."])

    def test_multinli_dataset_prepare_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = MultiNLIDataset(config=MultiNLIDatasetConfig(max_vectors=4))
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(dataset.archive_path, "w") as archive:
                    archive.writestr(
                        "multinli_1.0/multinli_1.0_train.jsonl",
                        "\n".join(
                            [
                                json.dumps({"sentence1": "People are talking.", "sentence2": "Humans converse."}),
                                json.dumps({"sentence1": "People are talking.", "sentence2": "Silence fills the room."}),
                            ]
                        )
                        + "\n",
                    )
                    archive.writestr(
                        "multinli_1.0/multinli_1.0_dev_matched.jsonl",
                        json.dumps({"sentence1": "Lights are bright.", "sentence2": "The room glows."}) + "\n",
                    )
                    archive.writestr(
                        "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
                        json.dumps({"sentence1": "Cars move fast.", "sentence2": "Vehicles accelerate."}) + "\n",
                    )

                with mock.patch.object(dataset, "build_encoder", return_value=FakeTextEncoder("fake-multinli")):
                    artifact_dir = dataset.ensure_prepared(download=False)
                    self.assertTrue(artifact_dir.exists())
                    embedding_matrix = dataset.load_embedding_matrix(download=False)

                self.assertEqual(embedding_matrix.num_embeddings, 4)
                self.assertEqual(embedding_matrix.embedding_dim, 4)
                self.assertEqual(embedding_matrix.metadata["encoder_name"], "fake-multinli")
                self.assertEqual(
                    embedding_matrix.texts,
                    [
                        "People are talking.",
                        "Humans converse.",
                        "Silence fills the room.",
                        "Lights are bright.",
                    ],
                )

    def test_load_dataset_returns_encoder_backed_sentence_datasets(self) -> None:
        snli = load_dataset("snli", max_vectors=10)
        multinli = load_dataset("multinli", max_vectors=10)
        self.assertIsInstance(snli, SNLIDataset)
        self.assertIsInstance(multinli, MultiNLIDataset)

    def test_flickr30k_dataset_prepare_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.dict("os.environ", {"AUTOENCODERS_CACHE": str(root)}, clear=False):
                dataset = Flickr30kDataset(config=Flickr30kDatasetConfig(max_vectors=5, clip_modality="both"))
                dataset.raw_dir.mkdir(parents=True, exist_ok=True)
                dataset.images_dir.mkdir(parents=True, exist_ok=True)

                image_a = dataset.images_dir / "1000.jpg"
                image_b = dataset.images_dir / "1001.jpg"
                image_a.write_bytes(b"fake-image-a")
                image_b.write_bytes(b"fake-image-b")
                dataset.manifest_path.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "image_id": "1000",
                                    "filename": "1000.jpg",
                                    "captions": ["a child smiles", "a kid is happy"],
                                }
                            ),
                            json.dumps(
                                {
                                    "image_id": "1001",
                                    "filename": "1001.jpg",
                                    "captions": ["a dog runs", "a pet sprints"],
                                }
                            ),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                with mock.patch.object(dataset, "build_encoder", return_value=FakeCLIPEncoder()):
                    artifact_dir = dataset.ensure_prepared(download=False)
                    self.assertTrue(artifact_dir.exists())
                    embedding_matrix = dataset.load_embedding_matrix(download=False)

                self.assertEqual(embedding_matrix.num_embeddings, 5)
                self.assertEqual(embedding_matrix.embedding_dim, 4)
                self.assertEqual(embedding_matrix.tokens[:2], ["image:1000", "image:1001"])
                self.assertEqual(embedding_matrix.metadata["encoder_name"], "fake-clip")
                self.assertEqual(embedding_matrix.metadata["encoder_pretrained"], "fake-pretrained")
                self.assertEqual(embedding_matrix.metadata["modality"], "both")

    def test_load_dataset_returns_flickr30k(self) -> None:
        dataset = load_dataset("flickr30k", max_vectors=10)
        self.assertIsInstance(dataset, Flickr30kDataset)

    def test_flickr30k_dataset_uses_fixed_hf_source(self) -> None:
        dataset = Flickr30kDataset(config=Flickr30kDatasetConfig(max_vectors=10))
        self.assertEqual(dataset.hf_dataset_name, "AnyModal/flickr30k")

    def test_flickr30k_extract_captions_uses_original_alt_text(self) -> None:
        self.assertEqual(
            Flickr30kDataset._extract_captions({"original_alt_text": ["first caption", "second caption"]}),
            ["first caption", "second caption"],
        )

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
