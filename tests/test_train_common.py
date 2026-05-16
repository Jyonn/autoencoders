"""Tests for training entrypoint helpers."""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from autoencoders import (
    VariationalAutoencoderConfig,
    VariationalAutoencoderModel,
    VectorQuantizedAutoencoderConfig,
    VectorQuantizedAutoencoderModel,
    build_mlp_backbone_kwargs,
)
from autoencoders.data.base import DatasetLoaders


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
TRAIN_COMMON_PATH = EXAMPLES_DIR / "_train_common.py"
TRAIN_COMMON_SPEC = importlib.util.spec_from_file_location("train_common", TRAIN_COMMON_PATH)
train_common = importlib.util.module_from_spec(TRAIN_COMMON_SPEC)
assert TRAIN_COMMON_SPEC.loader is not None
TRAIN_COMMON_SPEC.loader.exec_module(train_common)

CLI_PATH = EXAMPLES_DIR / "_cli.py"
CLI_SPEC = importlib.util.spec_from_file_location("train_cli", CLI_PATH)
train_cli = importlib.util.module_from_spec(CLI_SPEC)
assert CLI_SPEC.loader is not None
CLI_SPEC.loader.exec_module(train_cli)


class TrainCommonTest(unittest.TestCase):
    def test_parse_config_arguments_supports_dotted_model_and_encoder_options(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=["ae"], default="ae")
        parser.add_argument("--encoder", default="mlp")
        parser.add_argument("--decoder", default=None)

        argv = [
            "train_ae.py",
            "--model",
            "ae",
            "--model.latent_dim",
            "8",
            "--encoder.hidden_dims",
            "[32, 16]",
            "--encoder.activation",
            "gelu",
        ]
        with patch.object(sys, "argv", argv):
            args = train_cli.parse_config_arguments(
                parser,
                default_dataset_config={"dim": None, "max_vectors": None, "encoder": None, "encoder_batch_size": 128, "clip_pretrained": "laion2b", "clip_device": None, "clip_modality": "both"},
                default_trainer_config={},
                default_model_config={"latent_dim": 16, "reconstruction_loss": "mse"},
                default_encoder="mlp",
                default_encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
            )

        self.assertEqual(args.resolved_configs.model_config["latent_dim"], 8)
        self.assertEqual(args.resolved_configs.encoder_config["hidden_dims"], [32, 16])
        self.assertEqual(args.resolved_configs.encoder_config["activation"], "gelu")

    def test_parse_config_arguments_rejects_legacy_hidden_dims_flag(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=["ae"], default="ae")
        parser.add_argument("--encoder", default="mlp")
        parser.add_argument("--decoder", default=None)

        argv = [
            "train_ae.py",
            "--model",
            "ae",
            "--hidden-dims",
            "128",
            "64",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(ValueError):
                train_cli.parse_config_arguments(
                    parser,
                    default_dataset_config={"dim": None, "max_vectors": None, "encoder": None, "encoder_batch_size": 128, "clip_pretrained": "laion2b", "clip_device": None, "clip_modality": "both"},
                    default_trainer_config={},
                    default_model_config={"latent_dim": 16, "reconstruction_loss": "mse"},
                    default_encoder="mlp",
                    default_encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
                )

    def test_parse_config_arguments_supports_dotted_dataset_options(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", choices=["snli"], default="snli")
        parser.add_argument("--model", choices=["ae"], default="ae")
        parser.add_argument("--encoder", default="mlp")
        parser.add_argument("--decoder", default=None)

        argv = [
            "train_ae.py",
            "--dataset",
            "snli",
            "--model",
            "ae",
            "--dataset.encoder",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--dataset.encoder_batch_size",
            "64",
        ]
        with patch.object(sys, "argv", argv):
            args = train_cli.parse_config_arguments(
                parser,
                default_dataset_config={"dim": None, "max_vectors": None, "encoder": None, "encoder_batch_size": 128, "clip_pretrained": "laion2b", "clip_device": None, "clip_modality": "both"},
                default_trainer_config={},
                default_model_config={"latent_dim": 16, "reconstruction_loss": "mse"},
                default_encoder="mlp",
                default_encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
            )

        self.assertEqual(args.resolved_configs.dataset_config["encoder"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(args.resolved_configs.dataset_config["encoder_batch_size"], 64)

    def test_print_training_overview_includes_vae_specific_parameters(self) -> None:
        args = argparse.Namespace(
            dataset="glove",
            model="vae",
            output_dir="artifacts/test-vae",
            encoder=None,
            decoder=None,
            validation_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            epochs=5,
            patience=None,
            batch_size=256,
            learning_rate=1e-3,
            device="cpu",
            show_only_best_epochs=True,
            advice=True,
            resolved_configs=train_cli.ResolvedConfigArguments(
                dataset_config={
                    "dim": 50,
                    "max_vectors": 50000,
                    "encoder": None,
                    "encoder_batch_size": 128,
                    "clip_pretrained": "laion2b_s34b_b79k",
                    "clip_device": None,
                    "clip_modality": "both",
                },
                model_config={},
                encoder_config=None,
                decoder_config=None,
                trainer_config={},
            ),
            latent_dim=8,
        )
        config = VariationalAutoencoderConfig(
            input_dim=50,
            latent_dim=8,
            kl_weight=0.1,
            free_bits=0.02,
            kl_warmup_epochs=20,
            kl_start_weight=0.0,
        )
        model = VariationalAutoencoderModel(
            config,
            **build_mlp_backbone_kwargs([16, 8]),
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train_common.print_training_overview(args, model, input_dim=50)
        output = buffer.getvalue()

        self.assertIn("TRAIN PLAN", output)
        self.assertIn("Command", output)
        self.assertIn("kl_weight", output)
        self.assertIn("free_bits", output)
        self.assertIn("advice", output)
        self.assertIn("Encoder (mlp)", output)
        self.assertIn("hidden_dims", output)

    def test_print_training_overview_includes_vq_specific_parameters(self) -> None:
        args = argparse.Namespace(
            dataset="glove",
            model="vqvae",
            output_dir="artifacts/test-vq",
            encoder=None,
            decoder=None,
            validation_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            epochs=5,
            patience=None,
            batch_size=256,
            learning_rate=1e-3,
            device="cpu",
            show_only_best_epochs=True,
            advice=True,
            resolved_configs=train_cli.ResolvedConfigArguments(
                dataset_config={
                    "dim": 50,
                    "max_vectors": 50000,
                    "encoder": None,
                    "encoder_batch_size": 128,
                    "clip_pretrained": "laion2b_s34b_b79k",
                    "clip_device": None,
                    "clip_modality": "both",
                },
                model_config={},
                encoder_config=None,
                decoder_config=None,
                trainer_config={},
            ),
            latent_dim=8,
        )
        config = VectorQuantizedAutoencoderConfig(
            input_dim=50,
            latent_dim=8,
            codebook_size=64,
            commitment_weight=0.25,
            codebook_weight=1.0,
            use_ema_codebook=True,
            dead_code_reset=True,
        )
        model = VectorQuantizedAutoencoderModel(
            config,
            **build_mlp_backbone_kwargs([16, 8]),
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train_common.print_training_overview(args, model, input_dim=50)
        output = buffer.getvalue()

        self.assertIn("codebook_size", output)
        self.assertIn("use_ema_codebook", output)
        self.assertIn("dead_code_reset", output)
        self.assertIn("Decoder (mlp)", output)

    def test_validate_model_input_compatibility_prints_friendly_message(self) -> None:
        args = argparse.Namespace(dataset="glove", model="vqvae")
        config = VectorQuantizedAutoencoderConfig(
            input_dim=50,
            latent_dim=8,
            codebook_size=64,
        )
        model = VectorQuantizedAutoencoderModel(config, **build_mlp_backbone_kwargs([16, 8]))
        single_vector_batch = torch.randn(6, 50)
        dataloader = torch.utils.data.DataLoader(single_vector_batch, batch_size=2)
        loaders = DatasetLoaders(train=dataloader, validation=dataloader, test=dataloader)

        buffer = io.StringIO()
        with self.assertRaises(SystemExit) as context:
            with redirect_stdout(buffer):
                train_common.validate_model_input_compatibility(args, model, loaders)

        output = buffer.getvalue()
        self.assertEqual(context.exception.code, 2)
        self.assertIn("INPUT MISMATCH", output)
        self.assertIn("vqvae cannot train on glove", output)
        self.assertIn("B x T x D", output)
        self.assertIn("pqvae", output)


if __name__ == "__main__":
    unittest.main()
