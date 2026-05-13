"""Tests for the autoencoder trainer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from autoencoders import AutoencoderConfig, AutoencoderModel, AutoencoderTrainer, TrainingArguments
from autoencoders.data import DatasetLoaders, EmbeddingMatrix, EmbeddingTensorDataset


def build_dataset_loaders() -> DatasetLoaders:
    matrix = torch.randn(24, 8)
    embedding_matrix = EmbeddingMatrix(
        tokens=[f"token_{index}" for index in range(24)],
        matrix=matrix,
        token_to_index={f"token_{index}": index for index in range(24)},
    )
    dataset = EmbeddingTensorDataset(embedding_matrix)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    return DatasetLoaders(train=train_loader, validation=validation_loader, test=test_loader)


class AutoencoderTrainerTest(unittest.TestCase):
    def test_trainer_fits_and_saves_outputs(self) -> None:
        config = AutoencoderConfig(input_dim=8, latent_dim=4, hidden_dims=[6])
        model = AutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                epochs=2,
                learning_rate=1e-3,
                batch_size=6,
                device="cpu",
                seed=123,
            )
            trainer = AutoencoderTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders, metadata={"dataset": "dummy", "model": "ae"})

            self.assertIn("best_validation_loss", metrics)
            self.assertIn("final_test_loss", metrics)
            self.assertEqual(len(metrics["history"]), 2)
            self.assertEqual(metrics["dataset"], "dummy")
            self.assertEqual(metrics["model"], "ae")

            best_dir = Path(tmpdir) / "best"
            final_dir = Path(tmpdir) / "final"
            metrics_path = Path(tmpdir) / "metrics.json"
            self.assertTrue((best_dir / "config.json").exists())
            self.assertTrue((final_dir / "pytorch_model.bin").exists())
            self.assertTrue(metrics_path.exists())

            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_metrics["training_args"]["epochs"], 2)
            self.assertEqual(saved_metrics["dataset"], "dummy")


if __name__ == "__main__":
    unittest.main()
