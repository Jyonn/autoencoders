"""Tests for the finite scalar quantized autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import build_mlp_backbone_kwargs_from_model_config, FiniteScalarQuantizedAutoencoderConfig, FiniteScalarQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class FiniteScalarQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = FiniteScalarQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            num_levels=8,
            commitment_weight=0.25,
        )

    def test_forward_returns_scalar_quantized_indices(self) -> None:
        model = FiniteScalarQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 4))
        self.assertIn("commitment_loss", outputs.loss_dict)

    def test_export_includes_levels(self) -> None:
        model = FiniteScalarQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.extras["num_levels"], 8)
        self.assertEqual(tuple(artifact.extras["levels"].shape), (8,))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = FiniteScalarQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = FiniteScalarQuantizedAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.num_levels, 8)


if __name__ == "__main__":
    unittest.main()
