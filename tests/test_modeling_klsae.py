"""Tests for the KL sparse autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import build_mlp_backbone_kwargs_from_model_config, KLSparseAutoencoderConfig, KLSparseAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class KLSparseAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = KLSparseAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            sparsity_weight=0.1,
            target_activation=0.05,
        )

    def test_forward_returns_kl_sparsity_loss(self) -> None:
        model = KLSparseAutoencoderModel(self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        outputs = model(inputs=self.inputs)
        self.assertIn("kl_sparsity_loss", outputs.loss_dict)
        expected_loss = outputs.reconstruction_loss + self.config.sparsity_weight * outputs.kl_sparsity_loss
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = KLSparseAutoencoderModel(self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = KLSparseAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.target_activation, 0.05)


if __name__ == "__main__":
    unittest.main()
