"""Tests for training entrypoint helpers."""

from __future__ import annotations

import argparse
import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from autoencoders import VariationalAutoencoderConfig, VariationalAutoencoderModel, VectorQuantizedAutoencoderConfig, VectorQuantizedAutoencoderModel


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
TRAIN_COMMON_PATH = EXAMPLES_DIR / "_train_common.py"
TRAIN_COMMON_SPEC = importlib.util.spec_from_file_location("train_common", TRAIN_COMMON_PATH)
train_common = importlib.util.module_from_spec(TRAIN_COMMON_SPEC)
assert TRAIN_COMMON_SPEC.loader is not None
TRAIN_COMMON_SPEC.loader.exec_module(train_common)


class TrainCommonTest(unittest.TestCase):
    def test_print_training_overview_includes_vae_specific_parameters(self) -> None:
        args = argparse.Namespace(
            dataset="glove",
            model="vae",
            output_dir="artifacts/test-vae",
            dim=50,
            max_vectors=50000,
            encoder=None,
            encoder_batch_size=128,
            clip_pretrained="laion2b_s34b_b79k",
            clip_device=None,
            clip_modality="both",
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
            latent_dim=8,
            hidden_dims=[16, 8],
            activation="relu",
            reconstruction_loss="mse",
            kl_weight=0.1,
            beta=4.0,
            top_latent_dim=None,
            mmd_weight=5.0,
            mmd_bandwidths=[0.1, 0.2, 0.5],
            num_pseudo_inputs=128,
            pseudo_input_std=0.01,
            tc_weight=10.0,
            discriminator_hidden_dims=[128, 64],
            discriminator_learning_rate=None,
            discriminator_steps=1,
            kl_warmup_epochs=20,
            kl_start_weight=0.0,
            free_bits=0.02,
            noise_type="gaussian",
            noise_std=0.1,
            masking_ratio=0.3,
            apply_noise_in_eval=False,
        )
        model = VariationalAutoencoderModel(
            VariationalAutoencoderConfig(
                input_dim=50,
                latent_dim=8,
                hidden_dims=[16, 8],
                kl_weight=0.1,
                free_bits=0.02,
                kl_warmup_epochs=20,
                kl_start_weight=0.0,
            )
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

    def test_print_training_overview_includes_vq_specific_parameters(self) -> None:
        args = argparse.Namespace(
            dataset="glove",
            model="vqvae",
            output_dir="artifacts/test-vq",
            dim=50,
            max_vectors=50000,
            encoder=None,
            encoder_batch_size=128,
            clip_pretrained="laion2b_s34b_b79k",
            clip_device=None,
            clip_modality="both",
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
            latent_dim=8,
            hidden_dims=[16, 8],
            activation="relu",
            reconstruction_loss="mse",
            codebook_size=64,
            num_levels=8,
            num_codebooks=2,
            num_quantizers=2,
            commitment_weight=0.25,
            codebook_weight=1.0,
            use_ema_codebook=True,
            ema_decay=0.99,
            ema_epsilon=1e-5,
            dead_code_reset=True,
            dead_code_threshold=0,
        )
        model = VectorQuantizedAutoencoderModel(
            VectorQuantizedAutoencoderConfig(
                input_dim=50,
                latent_dim=8,
                hidden_dims=[16, 8],
                codebook_size=64,
                commitment_weight=0.25,
                codebook_weight=1.0,
                use_ema_codebook=True,
                dead_code_reset=True,
            )
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train_common.print_training_overview(args, model, input_dim=50)
        output = buffer.getvalue()

        self.assertIn("codebook_size", output)
        self.assertIn("use_ema_codebook", output)
        self.assertIn("dead_code_reset", output)


if __name__ == "__main__":
    unittest.main()
