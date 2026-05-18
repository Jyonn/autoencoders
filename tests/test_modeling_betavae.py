"""Tests for the beta variational autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import BetaVariationalAutoencoderConfig, BetaVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class BetaVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = BetaVariationalAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[12, 8],
            beta=4.0,
        )

    def test_forward_uses_beta_weighted_kl_loss(self) -> None:
        model = BetaVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        model.train()

        outputs = model(inputs=self.inputs)

        expected_loss = (
            outputs.reconstruction_loss + self.config.beta * outputs.free_bits_kl_loss
        )
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))
        self.assertIsNotNone(outputs.effective_kl_weight)

    def test_beta_is_saved_in_config(self) -> None:
        model = BetaVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = BetaVariationalAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.beta, 4.0)
        self.assertEqual(loaded.config.kl_weight, 4.0)


if __name__ == "__main__":
    unittest.main()
