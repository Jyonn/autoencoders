"""Tests for the adversarial autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

import torch

from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
from autoencoders import AdversarialAutoencoderConfig, AdversarialAutoencoderModel


class AdversarialAutoencoderModelTest(unittest.TestCase):
    def test_forward_returns_adversarial_metrics(self) -> None:
        config = AdversarialAutoencoderConfig(
            input_dim=6,
            latent_dim=3,
            hidden_dims=[5],
            discriminator_hidden_dims=[4],
            adversarial_weight=0.5,
        )
        model = AdversarialAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        inputs = torch.randn(4, 6)

        outputs = model(inputs=inputs)

        self.assertEqual(outputs.reconstruction.shape, inputs.shape)
        self.assertEqual(outputs.latents.shape, (4, 3))
        self.assertIsNotNone(outputs.adversarial_loss)
        self.assertIsNotNone(outputs.discriminator_loss)
        self.assertIn("adversarial_loss", outputs.loss_dict)
        self.assertIn("discriminator_loss", outputs.loss_dict)

    def test_export_uses_base_continuous_latent_contract(self) -> None:
        config = AdversarialAutoencoderConfig(input_dim=6, latent_dim=3, hidden_dims=[5], discriminator_hidden_dims=[4])
        model = AdversarialAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        inputs = torch.randn(2, 6)

        artifact = model.export(inputs, metadata={"split": "validation"})

        self.assertEqual(artifact.model_type, "adversarial_autoencoder")
        self.assertEqual(artifact.metadata["split"], "validation")
        self.assertEqual(tuple(artifact.latents.shape), (2, 3))

    def test_save_and_load_round_trip(self) -> None:
        config = AdversarialAutoencoderConfig(
            input_dim=6,
            latent_dim=3,
            hidden_dims=[5],
            adversarial_weight=0.75,
            discriminator_hidden_dims=[7, 5],
        )
        model = AdversarialAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = AdversarialAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(reloaded.config.adversarial_weight, 0.75)
        self.assertEqual(reloaded.config.discriminator_hidden_dims, [7, 5])


if __name__ == "__main__":
    unittest.main()
