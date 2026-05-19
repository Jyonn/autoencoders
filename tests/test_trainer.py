"""Tests for the autoencoder trainer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
from autoencoders import (
    AETrainer,
    AdversarialAutoencoderConfig,
    AdversarialAutoencoderModel,
    AdversarialAutoencoderTrainer,
    AdversarialAutoencoderTrainingArguments,
    AutoencoderConfig,
    AutoencoderModel,
    ContractiveAutoencoderConfig,
    ContractiveAutoencoderModel,
    DenoisingVariationalAutoencoderConfig,
    DenoisingVariationalAutoencoderModel,
    FiniteScalarQuantizedAutoencoderConfig,
    FiniteScalarQuantizedAutoencoderModel,
    HierarchicalVariationalAutoencoderConfig,
    HierarchicalVariationalAutoencoderModel,
    KLSparseAutoencoderConfig,
    KLSparseAutoencoderModel,
    TopKSparseAutoencoderConfig,
    TopKSparseAutoencoderModel,
    ProductQuantizedAutoencoderConfig,
    ProductQuantizedAutoencoderModel,
    ResidualQuantizedAutoencoderConfig,
    ResidualQuantizedAutoencoderModel,
    TrainerDisplay,
    TrainerDisplayConfig,
    TrainingArguments,
    VAETrainer,
    VariationalAutoencoderConfig,
    VariationalAutoencoderModel,
    VQTrainer,
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


def build_sequence_dataset_loaders() -> DatasetLoaders:
    class TensorSequenceDataset(torch.utils.data.Dataset):
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def __len__(self) -> int:
            return int(self.tensor.shape[0])

        def __getitem__(self, index: int) -> torch.Tensor:
            return self.tensor[index]

    tensor = torch.randn(24, 5, 8)
    dataset = TensorSequenceDataset(tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    return DatasetLoaders(train=train_loader, validation=validation_loader, test=test_loader)


class AutoencoderTrainerTest(unittest.TestCase):
    def test_trainer_fits_and_saves_outputs(self) -> None:
        config = AutoencoderConfig(latent_dim=4, hidden_dims=[6])
        model = AutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
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
            trainer = AETrainer(model=model, args=args)
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
            self.assertEqual(saved_metrics["advice"], [])

    def test_trainer_keeps_advice_disabled_by_default(self) -> None:
        config = AutoencoderConfig(latent_dim=4, hidden_dims=[6])
        model = AutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AETrainer(
                model=model,
                args=TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu"),
            )
            metrics = trainer.fit(dataloaders)

            self.assertEqual(metrics["advice"], [])

    def test_epochs_zero_requires_patience(self) -> None:
        with self.assertRaisesRegex(ValueError, "patience must be provided"):
            TrainingArguments(output_dir="unused", epochs=0)

    def test_training_arguments_validate_optimizer_scheduler_and_clip_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer_name"):
            TrainingArguments(output_dir="unused", optimizer_name="bogus")
        with self.assertRaisesRegex(ValueError, "lr_scheduler_type"):
            TrainingArguments(output_dir="unused", lr_scheduler_type="bogus")
        with self.assertRaisesRegex(ValueError, "grad_clip_norm"):
            TrainingArguments(output_dir="unused", grad_clip_norm=0.0)
        with self.assertRaisesRegex(ValueError, "epochs must be finite"):
            TrainingArguments(output_dir="unused", epochs=0, patience=1, lr_scheduler_type="linear")

    def test_trainer_supports_optimizer_scheduler_and_grad_clip(self) -> None:
        config = AutoencoderConfig(latent_dim=4, hidden_dims=[6])
        model = AutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                epochs=2,
                learning_rate=1e-3,
                optimizer_name="adamw",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                warmup_epochs=1,
                grad_clip_norm=1.0,
                batch_size=6,
                device="cpu",
            )
            trainer = AETrainer(model=model, args=args)
            trainer.fit(dataloaders)

            self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
            self.assertLess(trainer.optimizer.param_groups[0]["lr"], args.learning_rate)

    def test_trainer_stops_early_when_patience_is_reached(self) -> None:
        config = AutoencoderConfig(latent_dim=4, hidden_dims=[6])
        model = AutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        class ScriptedTrainer(AETrainer):
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

    def test_trainer_display_config_validates_progress_width(self) -> None:
        display = TrainerDisplayConfig()
        self.assertEqual(display.progress_width, 18)
        self.assertEqual(display.separator, " • ")
        self.assertIsInstance(TrainerDisplay(display), TrainerDisplay)

        with self.assertRaisesRegex(ValueError, "progress_width"):
            TrainerDisplayConfig(progress_width=0)

    def test_base_trainer_tracks_effective_kl_weight_for_vae_models(self) -> None:
        config = VariationalAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            kl_weight=0.8,
            kl_warmup_epochs=3,
            kl_start_weight=0.0,
        )
        model = VariationalAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                epochs=3,
                patience=None,
                learning_rate=1e-3,
                batch_size=6,
                device="cpu",
                seed=123,
            )
            trainer = VAETrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            history = metrics["history"]
            self.assertEqual(len(history), 3)
            self.assertAlmostEqual(history[0]["kl_weight"], 0.0, places=6)
            self.assertAlmostEqual(history[1]["kl_weight"], 0.4, places=6)
            self.assertAlmostEqual(history[2]["kl_weight"], 0.8, places=6)
            self.assertGreaterEqual(history[0]["train_free_bits_kl_loss"] + 1e-7, history[0]["train_kl_loss"])
            self.assertGreaterEqual(history[1]["train_free_bits_kl_loss"] + 1e-7, history[1]["train_kl_loss"])
            self.assertGreaterEqual(history[2]["train_free_bits_kl_loss"] + 1e-7, history[2]["train_kl_loss"])

    def test_base_trainer_applies_free_bits_floor_for_vae_models(self) -> None:
        config = VariationalAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            kl_weight=0.5,
            free_bits=0.25,
        )
        model = VariationalAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        args = TrainingArguments(
            output_dir="unused",
            epochs=1,
            device="cpu",
        )
        trainer = VAETrainer(model=model, args=args)

        outputs = model(inputs=torch.zeros(4, 8), current_epoch=1)
        effective_kl_loss = outputs.free_bits_kl_loss

        self.assertGreaterEqual(float(effective_kl_loss.item()), 1.0)

    def test_vae_trainer_emits_advice_for_posterior_collapse(self) -> None:
        config = VariationalAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            kl_weight=0.5,
            free_bits=0.02,
            kl_warmup_epochs=5,
        )
        model = VariationalAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        class ScriptedVAETrainer(VAETrainer):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.eval_calls = 0

            def train_epoch(self, dataloader) -> dict[str, float]:
                return {
                    "loss": 0.45,
                    "reconstruction_loss": 0.44,
                    "kl_loss": 0.000001,
                    "free_bits_kl_loss": 0.08,
                }

            def evaluate(self, dataloader) -> dict[str, float]:
                self.eval_calls += 1
                if self.eval_calls == 1:
                    return {
                        "loss": 0.46,
                        "reconstruction_loss": 0.45,
                        "kl_loss": 0.000001,
                        "free_bits_kl_loss": 0.08,
                    }
                return {
                    "loss": 0.47,
                    "reconstruction_loss": 0.46,
                    "kl_loss": 0.000001,
                    "free_bits_kl_loss": 0.08,
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ScriptedVAETrainer(
                model=model,
                args=TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu", advice=True),
            )
            metrics = trainer.fit(dataloaders)

            self.assertTrue(metrics["advice"])
            self.assertTrue(any("posterior collapse" in suggestion for suggestion in metrics["advice"]))
            self.assertTrue(any("free_bits" in suggestion for suggestion in metrics["advice"]))

    def test_base_trainer_tracks_contractive_loss_in_eval_for_cae_models(self) -> None:
        config = ContractiveAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            contractive_weight=0.1,
        )
        model = ContractiveAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu")
            trainer = AETrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("validation_contractive_loss", metrics["history"][0])
            self.assertGreaterEqual(metrics["history"][0]["validation_contractive_loss"], 0.0)

    def test_wasserstein_model_trains_with_base_trainer(self) -> None:
        config = WassersteinAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            mmd_weight=2.0,
        )
        model = WassersteinAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu")
            trainer = AETrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("train_mmd_loss", metrics["history"][0])
            self.assertIn("validation_mmd_loss", metrics["history"][0])

    def test_topk_and_kl_sparse_models_train_with_base_trainer(self) -> None:
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            topk_model = TopKSparseAutoencoderModel(
                config=TopKSparseAutoencoderConfig(latent_dim=4, hidden_dims=[6], topk=2),
                **build_mlp_backbone_kwargs_from_model_config(
                    TopKSparseAutoencoderConfig(latent_dim=4, hidden_dims=[6], topk=2),
                    feature_dim=8,
                ),
            )
            topk_metrics = AETrainer(
                model=topk_model,
                args=TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu"),
            ).fit(dataloaders)
            self.assertIn("train_topk_sparsity", topk_metrics["history"][0])

        with tempfile.TemporaryDirectory() as tmpdir:
            klsae_model = KLSparseAutoencoderModel(
                config=KLSparseAutoencoderConfig(
                    latent_dim=4,
                    hidden_dims=[6],
                    sparsity_weight=0.1,
                    target_activation=0.05,
                ),
                **build_mlp_backbone_kwargs_from_model_config(
                    KLSparseAutoencoderConfig(
                        latent_dim=4,
                        hidden_dims=[6],
                        sparsity_weight=0.1,
                        target_activation=0.05,
                    ),
                    feature_dim=8,
                ),
            )
            klsae_metrics = AETrainer(
                model=klsae_model,
                args=TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu"),
            ).fit(dataloaders)
            self.assertIn("train_kl_sparsity_loss", klsae_metrics["history"][0])

    def test_dvae_and_hvae_train_with_base_trainer(self) -> None:
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            dvae_config = DenoisingVariationalAutoencoderConfig(latent_dim=4, hidden_dims=[6], kl_weight=0.5)
            dvae_model = DenoisingVariationalAutoencoderModel(
                config=dvae_config,
                **build_mlp_backbone_kwargs_from_model_config(dvae_config, feature_dim=8),
            )
            dvae_metrics = VAETrainer(
                model=dvae_model,
                args=TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu"),
            ).fit(dataloaders)
            self.assertIn("train_kl_loss", dvae_metrics["history"][0])

        with tempfile.TemporaryDirectory() as tmpdir:
            hvae_config = HierarchicalVariationalAutoencoderConfig(
                latent_dim=4,
                top_latent_dim=3,
                hidden_dims=[6],
                kl_weight=0.5,
            )
            hvae_model = HierarchicalVariationalAutoencoderModel(
                config=hvae_config,
                **build_mlp_backbone_kwargs_from_model_config(hvae_config, feature_dim=8),
            )
            hvae_metrics = VAETrainer(
                model=hvae_model,
                args=TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu"),
            ).fit(dataloaders)
            self.assertIn("train_hierarchical_kl_loss", hvae_metrics["history"][0])

    def test_fsq_trains_with_quantized_trainer(self) -> None:
        config = FiniteScalarQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            num_levels=8,
        )
        model = FiniteScalarQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(output_dir=tmpdir, epochs=1, device="cpu")
            trainer = VQTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("train_active_codes", metrics["history"][0])
            self.assertIn("validation_codebook_usage_ratio", metrics["history"][0])
            self.assertLessEqual(metrics["history"][0]["validation_codebook_usage_ratio"], 1.0)

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
            latent_dim=4,
            hidden_dims=[6],
            discriminator_hidden_dims=[5],
            adversarial_weight=0.25,
        )
        model = AdversarialAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
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
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=16,
            dead_code_reset=True,
        )
        model = VectorQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_sequence_dataset_loaders()

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
            trainer = VQTrainer(model=model, args=args)
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

    def test_quantized_trainer_generates_codebook_advice(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=16,
            use_ema_codebook=False,
        )
        model = VectorQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        trainer = VQTrainer(model=model, args=TrainingArguments(output_dir="unused", device="cpu", advice=True))

        advice = trainer.generate_advice(
            {
                "history": [],
                "final_test_metrics": {
                    "reconstruction_loss": 1.0,
                    "codebook_usage_ratio": 0.25,
                    "commitment_loss": 0.35,
                    "codebook_loss": 0.4,
                },
            }
        )

        self.assertTrue(any("codebook_size" in suggestion for suggestion in advice))
        self.assertTrue(any("commitment_weight" in suggestion for suggestion in advice))
        self.assertTrue(any("codebook_weight" in suggestion for suggestion in advice))

    def test_quantized_trainer_computes_codebook_metrics(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=4,
        )
        model = VectorQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        args = TrainingArguments(output_dir="unused", device="cpu")
        trainer = VQTrainer(model=model, args=args)

        counts = torch.tensor([3, 1, 0, 0], dtype=torch.long)
        metrics = trainer.compute_codebook_metrics(counts)

        self.assertEqual(metrics["active_codes"], 2.0)
        self.assertEqual(metrics["codebook_size"], 4.0)
        self.assertAlmostEqual(metrics["codebook_usage_ratio"], 0.5, places=6)
        self.assertAlmostEqual(metrics["dead_code_ratio"], 0.5, places=6)
        self.assertGreater(metrics["codebook_perplexity"], 1.0)

    def test_quantized_trainer_computes_multi_codebook_metrics(self) -> None:
        config = ProductQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=4,
            num_codebooks=2,
        )
        model = ProductQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        args = TrainingArguments(output_dir="unused", device="cpu")
        trainer = VQTrainer(model=model, args=args)

        counts = torch.tensor([[3, 1, 0, 0], [0, 2, 2, 0]], dtype=torch.long)
        metrics = trainer.compute_codebook_metrics(counts)

        self.assertEqual(metrics["active_codes"], 4.0)
        self.assertEqual(metrics["codebook_size"], 8.0)
        self.assertAlmostEqual(metrics["codebook_usage_ratio"], 0.5, places=6)
        self.assertAlmostEqual(metrics["dead_code_ratio"], 0.5, places=6)
        self.assertGreater(metrics["codebook_perplexity"], 1.0)

    def test_quantized_trainer_tracks_multi_codebook_usage(self) -> None:
        config = ResidualQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=8,
            num_quantizers=2,
            dead_code_reset=True,
        )
        model = ResidualQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8))
        dataloaders = build_dataset_loaders()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                epochs=1,
                device="cpu",
            )
            trainer = VQTrainer(model=model, args=args)
            metrics = trainer.fit(dataloaders)

            self.assertIn("train_active_codes", metrics["history"][0])
            self.assertIn("validation_codebook_usage_ratio", metrics["history"][0])

    def test_quantized_trainer_extracts_collision_signatures(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[6],
            codebook_size=16,
        )
        model = VectorQuantizedAutoencoderModel(
            config=config,
            **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=8),
        )
        trainer = VQTrainer(model=model, args=TrainingArguments(output_dir="unused", device="cpu"))

        codebook_indices = torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
            ],
            dtype=torch.long,
        )

        signatures = trainer.extract_collision_signatures(codebook_indices)

        self.assertEqual(len(signatures), 2)
        self.assertIn((0, 1, 2, 3), signatures)
        self.assertIn((4, 5, 6, 7), signatures)

    def test_vq_config_validates_dead_code_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "dead_code_threshold"):
            VectorQuantizedAutoencoderConfig(
                latent_dim=4,
                hidden_dims=[6],
                codebook_size=16,
                dead_code_threshold=-1,
            )


if __name__ == "__main__":
    unittest.main()
