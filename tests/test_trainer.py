"""Tests for the autoencoder trainer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from autoencoders import (
    AdversarialAutoencoderConfig,
    AdversarialAutoencoderModel,
    AdversarialAutoencoderTrainer,
    AdversarialAutoencoderTrainingArguments,
    AutoencoderConfig,
    AutoencoderModel,
    AutoencoderTrainer,
    ContractiveAutoencoderConfig,
    ContractiveAutoencoderModel,
    ContractiveAutoencoderTrainer,
    QuantizedAutoencoderTrainer,
    QuantizedAutoencoderTrainingArguments,
    ProductQuantizedAutoencoderConfig,
    ProductQuantizedAutoencoderModel,
    ResidualQuantizedAutoencoderConfig,
    ResidualQuantizedAutoencoderModel,
    TrainerDisplayConfig,
    TrainingArguments,
    VAETrainer,
    VAETrainingArguments,
    VariationalAutoencoderConfig,
    VariationalAutoencoderModel,
    VectorQuantizedAutoencoderConfig,
    VectorQuantizedAutoencoderModel,
    WassersteinAutoencoderConfig,
    WassersteinAutoencoderModel,
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

    def test_trainer_display_config_validates_progress_width(self) -> None:
        display = TrainerDisplayConfig()
        self.assertEqual(display.progress_width, 18)
        self.assertEqual(display.separator, " • ")

        with self.assertRaisesRegex(ValueError, "progress_width"):
            TrainerDisplayConfig(progress_width=0)

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
            self.assertAlmostEqual(history[0]["train_free_bits_kl_loss"], history[0]["train_kl_loss"], places=3)
            self.assertAlmostEqual(history[1]["train_free_bits_kl_loss"], history[1]["train_kl_loss"], places=3)
            self.assertAlmostEqual(history[2]["train_free_bits_kl_loss"], history[2]["train_kl_loss"], places=3)

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

    def test_contractive_trainer_tracks_contractive_loss_in_eval(self) -> None:
        config = ContractiveAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            contractive_weight=0.1,
        )
        model = ContractiveAutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu")
            trainer = ContractiveAutoencoderTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("validation_contractive_loss", metrics["history"][0])
            self.assertGreaterEqual(metrics["history"][0]["validation_contractive_loss"], 0.0)

    def test_wasserstein_model_trains_with_base_trainer(self) -> None:
        config = WassersteinAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            mmd_weight=2.0,
        )
        model = WassersteinAutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu")
            trainer = AutoencoderTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("train_mmd_loss", metrics["history"][0])
            self.assertIn("validation_mmd_loss", metrics["history"][0])

    def test_adversarial_training_arguments_validate_fields(self) -> None:
        defaults = AdversarialAutoencoderTrainingArguments(output_dir="unused")
        self.assertEqual(defaults.discriminator_steps, 1)
        self.assertIsNone(defaults.generator_learning_rate)
        self.assertIsNone(defaults.discriminator_learning_rate)

        with self.assertRaisesRegex(ValueError, "generator_learning_rate"):
            AdversarialAutoencoderTrainingArguments(output_dir="unused", generator_learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "discriminator_learning_rate"):
            AdversarialAutoencoderTrainingArguments(output_dir="unused", discriminator_learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "discriminator_steps"):
            AdversarialAutoencoderTrainingArguments(output_dir="unused", discriminator_steps=0)

    def test_adversarial_trainer_tracks_generator_and_discriminator_metrics(self) -> None:
        config = AdversarialAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            discriminator_hidden_dims=[5],
            adversarial_weight=0.25,
        )
        model = AdversarialAutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = AdversarialAutoencoderTrainingArguments(
                output_dir=tmpdir,
                epochs=1,
                device="cpu",
            )
            trainer = AdversarialAutoencoderTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("train_adversarial_loss", metrics["history"][0])
            self.assertIn("train_discriminator_loss", metrics["history"][0])
            self.assertIn("validation_adversarial_loss", metrics["history"][0])
            self.assertIn("validation_discriminator_loss", metrics["history"][0])
            self.assertIn("adversarial_loss", metrics["final_test_metrics"])
            self.assertIn("discriminator_loss", metrics["final_test_metrics"])

    def test_quantized_trainer_tracks_codebook_metrics(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=16,
        )
        model = VectorQuantizedAutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = QuantizedAutoencoderTrainingArguments(
                output_dir=tmpdir,
                epochs=2,
                patience=None,
                learning_rate=1e-3,
                batch_size=6,
                device="cpu",
                seed=123,
            )
            trainer = QuantizedAutoencoderTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            history = metrics["history"]
            self.assertEqual(len(history), 2)
            self.assertIn("train_active_codes", history[0])
            self.assertIn("validation_active_codes", history[0])
            self.assertIn("train_codebook_perplexity", history[0])
            self.assertIn("validation_codebook_usage_ratio", history[0])
            self.assertIn("train_dead_code_reset_count", history[0])
            self.assertLessEqual(history[0]["train_active_codes"], 16.0)
            self.assertLessEqual(history[0]["validation_codebook_usage_ratio"], 1.0)
            self.assertGreaterEqual(history[0]["validation_dead_code_ratio"], 0.0)
            self.assertIn("active_codes", metrics["final_test_metrics"])
            self.assertIn("codebook_perplexity", metrics["final_test_metrics"])

    def test_quantized_trainer_computes_codebook_metrics(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=4,
        )
        model = VectorQuantizedAutoencoderModel(config)
        args = QuantizedAutoencoderTrainingArguments(output_dir="unused", device="cpu")
        trainer = QuantizedAutoencoderTrainer(model=model, args=args)

        counts = torch.tensor([3, 1, 0, 0], dtype=torch.long)
        metrics = trainer.compute_codebook_metrics(counts)

        self.assertEqual(metrics["active_codes"], 2.0)
        self.assertEqual(metrics["codebook_size"], 4.0)
        self.assertAlmostEqual(metrics["codebook_usage_ratio"], 0.5, places=6)
        self.assertAlmostEqual(metrics["dead_code_ratio"], 0.5, places=6)
        self.assertGreater(metrics["codebook_perplexity"], 1.0)

    def test_quantized_trainer_computes_multi_codebook_metrics(self) -> None:
        config = ProductQuantizedAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=4,
            num_codebooks=2,
        )
        model = ProductQuantizedAutoencoderModel(config)
        args = QuantizedAutoencoderTrainingArguments(output_dir="unused", device="cpu")
        trainer = QuantizedAutoencoderTrainer(model=model, args=args)

        counts = torch.tensor([[3, 1, 0, 0], [0, 2, 2, 0]], dtype=torch.long)
        metrics = trainer.compute_codebook_metrics(counts)

        self.assertEqual(metrics["active_codes"], 4.0)
        self.assertEqual(metrics["codebook_size"], 8.0)
        self.assertAlmostEqual(metrics["codebook_usage_ratio"], 0.5, places=6)
        self.assertAlmostEqual(metrics["dead_code_ratio"], 0.5, places=6)
        self.assertGreater(metrics["codebook_perplexity"], 1.0)

    def test_quantized_trainer_tracks_multi_codebook_usage(self) -> None:
        config = ResidualQuantizedAutoencoderConfig(
            input_dim=8,
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=8,
            num_quantizers=2,
        )
        model = ResidualQuantizedAutoencoderModel(config)
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = QuantizedAutoencoderTrainingArguments(
                output_dir=tmpdir,
                epochs=1,
                device="cpu",
            )
            trainer = QuantizedAutoencoderTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("train_active_codes", metrics["history"][0])
            self.assertIn("validation_codebook_usage_ratio", metrics["history"][0])
            self.assertLessEqual(metrics["history"][0]["validation_codebook_usage_ratio"], 1.0)

    def test_quantized_training_arguments_validate_dead_code_threshold(self) -> None:
        defaults = QuantizedAutoencoderTrainingArguments(output_dir="unused")
        self.assertTrue(defaults.dead_code_reset)
        self.assertEqual(defaults.dead_code_threshold, 0)

        with self.assertRaisesRegex(ValueError, "dead_code_threshold"):
            QuantizedAutoencoderTrainingArguments(output_dir="unused", dead_code_threshold=-1)


if __name__ == "__main__":
    unittest.main()
