"""Tests for the FactorVAE model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import FactorVariationalAutoencoderConfig, FactorVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class FactorVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(6, 16)
        self.config = FactorVariationalAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            kl_weight=0.2,
            tc_weight=5.0,
        )

    def test_forward_returns_expected_fields(self) -> None:
        model = FactorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.train()

        outputs = model(inputs=self.inputs, current_epoch=1)

        self.assertEqual(tuple(outputs.reconstruction.shape), (6, 16))
        self.assertEqual(tuple(outputs.latents.shape), (6, 4))
        self.assertIn("total_correlation_loss", outputs.loss_dict)
        self.assertIn("discriminator_loss", outputs.loss_dict)

    def test_permute_dims_preserves_shape(self) -> None:
        model = FactorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        latents = torch.randn(6, 4)
        permuted = model.permute_dims(latents)
        self.assertEqual(tuple(permuted.shape), (6, 4))

    def test_export_includes_posterior_statistics(self) -> None:
        model = FactorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.eval()
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "factor_variational_autoencoder")
        self.assertEqual(tuple(artifact.posterior_mean.shape), (6, 4))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = FactorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = FactorVariationalAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.tc_weight, 5.0)


if __name__ == "__main__":
    unittest.main()
