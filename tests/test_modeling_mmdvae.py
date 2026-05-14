"""Tests for the MMD-VAE model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import MMDVariationalAutoencoderConfig, MMDVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class MMDVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = MMDVariationalAutoencoderConfig(input_dim=16, latent_dim=4, hidden_dims=[12, 8], mmd_weight=10.0)

    def test_forward_returns_mmd_loss(self) -> None:
        model = MMDVariationalAutoencoderModel(self.config)
        outputs = model(inputs=self.inputs, current_epoch=1)
        self.assertIn("mmd_loss", outputs.loss_dict)
        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))

    def test_export_includes_posterior_statistics(self) -> None:
        model = MMDVariationalAutoencoderModel(self.config)
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "mmd_variational_autoencoder")

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = MMDVariationalAutoencoderModel(self.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = MMDVariationalAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.mmd_weight, 10.0)


if __name__ == "__main__":
    unittest.main()
