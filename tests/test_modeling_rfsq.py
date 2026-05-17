"""Tests for the residual FSQ model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import ResidualFiniteScalarQuantizedAutoencoderConfig, ResidualFiniteScalarQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class ResidualFiniteScalarQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = ResidualFiniteScalarQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            num_quantizers=3,
        )

    def test_forward_returns_stacked_code_indices(self) -> None:
        model = ResidualFiniteScalarQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 12))
        self.assertIn("quantization_residual_loss", outputs.loss_dict)

    def test_export_includes_level_metadata(self) -> None:
        model = ResidualFiniteScalarQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "residual_finite_scalar_quantized_autoencoder")
        self.assertEqual(artifact.extras["num_quantizers"], 3)

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = ResidualFiniteScalarQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = ResidualFiniteScalarQuantizedAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.num_quantizers, 3)


if __name__ == "__main__":
    unittest.main()
