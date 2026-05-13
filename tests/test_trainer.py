"""Tests for the autoencoder trainer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from autoencoders import (
    AutoencoderConfig,
    AutoencoderModel,
    AutoencoderTrainer,
    TrainingArguments,
    VAETrainer,
    VAETrainingArguments,
    VariationalAutoencoderConfig,
    VariationalAutoencoderModel,
)
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
                patience=None,
                learning_rate=1e-3,
                batch_size=6,
                device="cpu",
                seed=123,
            )
            trainer = AutoencoderTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders, metadata={"dataset": "dummy", "model": "ae"})

            self.assertIn("best_validation_loss", metrics)
            self.assertIn("best_epoch", metrics)
            self.assertIn("epochs_completed", metrics)
            self.assertIn("final_test_loss", metrics)
            self.assertIn("final_test_metrics", metrics)
            self.assertEqual(len(metrics["history"]), 2)
            self.assertEqual(metrics["epochs_completed"], 2)
            self.assertFalse(metrics["stopped_early"])
            self.assertEqual(metrics["dataset"], "dummy")
            self.assertEqual(metrics["model"], "ae")
            self.assertIn("train_loss", metrics["history"][0])
            self.assertIn("validation_loss", metrics["history"][0])

            best_dir = Path(tmpdir) / "best"
            final_dir = Path(tmpdir) / "final"
            metrics_path = Path(tmpdir) / "metrics.json"
            self.assertTrue((best_dir / "config.json").exists())
            self.assertTrue((final_dir / "pytorch_model.bin").exists())
            self.assertTrue(metrics_path.exists())

            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_metrics["training_args"]["epochs"], 2)
            self.assertIsNone(saved_metrics["training_args"]["patience"])
            self.assertEqual(saved_metrics["dataset"], "dummy")

    def test_epochs_zero_requires_patience(self) -> None:
        with self.assertRaisesRegex(ValueError, "patience must be provided"):
            TrainingArguments(output_dir="unused", epochs=0)

    def test_trainer_stops_early_when_patience_is_reached(self) -> None:
        config = AutoencoderConfig(input_dim=8, latent_dim=4, hidden_dims=[6])
        model = AutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        class ScriptedTrainer(AutoencoderTrainer):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.validation_losses = iter([5.0, 4.0, 4.5, 4.6])
                self.test_metrics = {"loss": 3.5}

            def train_epoch(self, dataloader) -> dict[str, float]:
                return {"loss": 1.0}

            def evaluate(self, dataloader) -> dict[str, float]:
                try:
                    loss = next(self.validation_losses)
                except StopIteration:
                    loss = self.test_metrics["loss"]
                return {"loss": loss}

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                epochs=0,
                patience=2,
                learning_rate=1e-3,
                batch_size=6,
                device="cpu",
                seed=123,
            )
            trainer = ScriptedTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertTrue(metrics["stopped_early"])
            self.assertEqual(metrics["best_epoch"], 2)
            self.assertEqual(metrics["epochs_completed"], 4)
            self.assertEqual(len(metrics["history"]), 4)
            self.assertEqual(metrics["final_test_loss"], 3.5)

    def test_vae_training_arguments_validate_warmup_fields(self) -> None:
        defaults = VAETrainingArguments(output_dir="unused")
        self.assertEqual(defaults.kl_warmup_epochs, 20)
        self.assertEqual(defaults.kl_start_weight, 0.0)
        self.assertEqual(defaults.free_bits, 0.02)

        with self.assertRaisesRegex(ValueError, "kl_warmup_epochs"):
            VAETrainingArguments(output_dir="unused", kl_warmup_epochs=-1)
        with self.assertRaisesRegex(ValueError, "kl_start_weight"):
            VAETrainingArguments(output_dir="unused", kl_start_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "free_bits"):
            VAETrainingArguments(output_dir="unused", free_bits=-0.1)

    def test_vae_trainer_tracks_effective_kl_weight(self) -> None:
        config = VariationalAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            kl_weight=0.8,
        )
        model = VariationalAutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = VAETrainingArguments(
                output_dir=tmpdir,
                epochs=3,
                patience=None,
                learning_rate=1e-3,
                batch_size=6,
                device="cpu",
                seed=123,
                kl_warmup_epochs=3,
                kl_start_weight=0.0,
            )
            trainer = VAETrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            history = metrics["history"]
            self.assertEqual(len(history), 3)
            self.assertAlmostEqual(history[0]["kl_weight"], 0.0, places=6)
            self.assertAlmostEqual(history[1]["kl_weight"], 0.4, places=6)
            self.assertAlmostEqual(history[2]["kl_weight"], 0.8, places=6)
            self.assertAlmostEqual(history[0]["train_free_bits_kl_loss"], history[0]["train_kl_loss"], places=6)
            self.assertAlmostEqual(history[1]["train_free_bits_kl_loss"], history[1]["train_kl_loss"], places=6)
            self.assertAlmostEqual(history[2]["train_free_bits_kl_loss"], history[2]["train_kl_loss"], places=6)

    def test_vae_trainer_applies_free_bits_floor(self) -> None:
        config = VariationalAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            kl_weight=0.5,
        )
        model = VariationalAutoencoderModel(config)
        args = VAETrainingArguments(
            output_dir="unused",
            epochs=1,
            device="cpu",
            free_bits=0.25,
        )
        trainer = VAETrainer(model=model, args=args)

        outputs = model(inputs=torch.zeros(4, 8))
        effective_kl_loss = trainer.compute_free_bits_kl_loss(outputs)

        self.assertGreaterEqual(float(effective_kl_loss.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
