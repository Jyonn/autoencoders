"""Tests for the Wasserstein autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

import torch

from autoencoders import WassersteinAutoencoderConfig, WassersteinAutoencoderModel


class WassersteinAutoencoderModelTest(unittest.TestCase):
    def test_forward_returns_mmd_loss(self) -> None:
        model = WassersteinAutoencoderModel(
            WassersteinAutoencoderConfig(
                input_dim=6,
                latent_dim=3,
                hidden_dims=[5],
                mmd_weight=5.0,
            )
        )
        inputs = torch.randn(4, 6)

        outputs = model(inputs=inputs)

        self.assertEqual(outputs.reconstruction.shape, inputs.shape)
        self.assertEqual(outputs.latents.shape, (4, 3))
        self.assertIsNotNone(outputs.mmd_loss)
        self.assertGreaterEqual(float(outputs.mmd_loss.item()), 0.0)
        self.assertIn("mmd_loss", outputs.loss_dict)

    def test_export_includes_latents_and_reconstruction(self) -> None:
        model = WassersteinAutoencoderModel(
            WassersteinAutoencoderConfig(input_dim=6, latent_dim=3, hidden_dims=[5])
        )
        inputs = torch.randn(2, 6)

        artifact = model.export(inputs)

        self.assertEqual(artifact.model_type, "wasserstein_autoencoder")
        self.assertEqual(tuple(artifact.latents.shape), (2, 3))
        self.assertEqual(tuple(artifact.reconstruction.shape), (2, 6))

    def test_save_and_load_round_trip(self) -> None:
        config = WassersteinAutoencoderConfig(
            input_dim=6,
            latent_dim=3,
            hidden_dims=[5],
            mmd_weight=7.5,
            mmd_bandwidths=[0.5, 1.0],
        )
        model = WassersteinAutoencoderModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = WassersteinAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(reloaded.config.mmd_weight, 7.5)
        self.assertEqual(reloaded.config.mmd_bandwidths, [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
