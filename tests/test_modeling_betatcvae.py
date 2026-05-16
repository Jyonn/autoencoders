"""Tests for the Beta-TCVAE model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import build_mlp_backbone_kwargs_from_model_config, BetaTCVariationalAutoencoderConfig, BetaTCVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class BetaTCVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(6, 16)
        self.config = BetaTCVariationalAutoencoderConfig(input_dim=16, latent_dim=4, hidden_dims=[12, 8])

    def test_forward_returns_decomposed_kl_terms(self) -> None:
        model = BetaTCVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        outputs = model(inputs=self.inputs, current_epoch=1)
        self.assertIn("mutual_information_loss", outputs.loss_dict)
        self.assertIn("total_correlation_loss", outputs.loss_dict)
        self.assertIn("dimension_wise_kl_loss", outputs.loss_dict)

    def test_export_includes_posterior_statistics(self) -> None:
        model = BetaTCVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "beta_tc_variational_autoencoder")
        self.assertEqual(tuple(artifact.posterior_logvar.shape), (6, 4))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = BetaTCVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = BetaTCVariationalAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.total_correlation_weight, 6.0)


if __name__ == "__main__":
    unittest.main()
