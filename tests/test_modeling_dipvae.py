"""Tests for the DIP-VAE model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import DIPVariationalAutoencoderConfig, DIPVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class DIPVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(5, 16)
        self.config = DIPVariationalAutoencoderConfig(latent_dim=4, hidden_dims=[12, 8], dip_weight=10.0)

    def test_forward_returns_dip_loss(self) -> None:
        model = DIPVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        outputs = model(inputs=self.inputs, current_epoch=1)
        self.assertEqual(tuple(outputs.reconstruction.shape), (5, 16))
        self.assertIn("dip_loss", outputs.loss_dict)

    def test_export_includes_posterior_statistics(self) -> None:
        model = DIPVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "dip_variational_autoencoder")
        self.assertEqual(tuple(artifact.posterior_mean.shape), (5, 4))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = DIPVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = DIPVariationalAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.dip_weight, 10.0)


if __name__ == "__main__":
    unittest.main()
