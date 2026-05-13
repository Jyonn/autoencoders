"""Tests for the contractive autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

import torch

from autoencoders import ContractiveAutoencoderConfig, ContractiveAutoencoderModel


class ContractiveAutoencoderModelTest(unittest.TestCase):
    def test_forward_returns_contractive_loss(self) -> None:
        model = ContractiveAutoencoderModel(
            ContractiveAutoencoderConfig(
                input_dim=6,
                latent_dim=3,
                hidden_dims=[5],
                contractive_weight=0.1,
            )
        )
        inputs = torch.randn(4, 6)

        outputs = model(inputs=inputs)

        self.assertEqual(outputs.reconstruction.shape, inputs.shape)
        self.assertEqual(outputs.latents.shape, (4, 3))
        self.assertIsNotNone(outputs.contractive_loss)
        self.assertGreaterEqual(float(outputs.contractive_loss.item()), 0.0)
        self.assertIn("contractive_loss", outputs.loss_dict)

    def test_export_works_under_no_grad(self) -> None:
        model = ContractiveAutoencoderModel(
            ContractiveAutoencoderConfig(input_dim=6, latent_dim=3, hidden_dims=[5])
        )
        inputs = torch.randn(2, 6)

        artifact = model.export(inputs, metadata={"split": "test"})

        self.assertEqual(artifact.model_type, "contractive_autoencoder")
        self.assertEqual(artifact.metadata["split"], "test")
        self.assertEqual(tuple(artifact.reconstruction.shape), (2, 6))

    def test_save_and_load_round_trip(self) -> None:
        config = ContractiveAutoencoderConfig(input_dim=6, latent_dim=3, hidden_dims=[5], contractive_weight=0.2)
        model = ContractiveAutoencoderModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = ContractiveAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(reloaded.config.contractive_weight, 0.2)
        self.assertEqual(reloaded.config.hidden_dims, [5])


if __name__ == "__main__":
    unittest.main()
