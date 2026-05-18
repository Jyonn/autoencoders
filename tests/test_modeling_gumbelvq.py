"""Tests for the Gumbel-VQ model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import GumbelQuantizedAutoencoderConfig, GumbelQuantizedAutoencoderModel
    from autoencoders.data import TensorSpec


@unittest.skipIf(torch is None, "torch is required for model tests")
class GumbelQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 5, 16)
        self.config = GumbelQuantizedAutoencoderConfig(latent_dim=4, hidden_dims=[12, 8], codebook_size=32)

    def test_forward_returns_assignment_metrics(self) -> None:
        model = GumbelQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 5))
        self.assertIn("assignment_entropy", outputs.loss_dict)

    def test_export_includes_codebook(self) -> None:
        model = GumbelQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "gumbel_quantized_autoencoder")
        self.assertEqual(tuple(artifact.extras["codebook"].shape), (32, 4))

    def test_gumbel_vq_requires_multi_vector_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank >= 2"):
            GumbelQuantizedAutoencoderModel(
                config=self.config,
                sample_spec=TensorSpec(shape=(16,)),
                encoder="mlp",
                decoder="mlp",
                encoder_config={"hidden_dims": [12, 8], "activation": "relu", "use_bias": True},
                decoder_config={"hidden_dims": [12, 16], "activation": "relu", "use_bias": True},
            )

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = GumbelQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = GumbelQuantizedAutoencoderModel.from_pretrained(tmpdir)
        self.assertTrue(loaded.config.straight_through)


if __name__ == "__main__":
    unittest.main()
