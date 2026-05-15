"""Tests for the hierarchical variational autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import build_mlp_backbone_kwargs_from_model_config, HierarchicalVariationalAutoencoderConfig, HierarchicalVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class HierarchicalVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = HierarchicalVariationalAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            top_latent_dim=3,
            hidden_dims=[12, 8],
            kl_weight=0.5,
        )

    def test_forward_returns_hierarchical_latents(self) -> None:
        model = HierarchicalVariationalAutoencoderModel(self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.train()
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 7))
        self.assertEqual(tuple(outputs.posterior_mean.shape), (4, 7))
        self.assertIn("hierarchical_kl_loss", outputs.loss_dict)

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = HierarchicalVariationalAutoencoderModel(self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = HierarchicalVariationalAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.top_latent_dim, 3)


if __name__ == "__main__":
    unittest.main()
