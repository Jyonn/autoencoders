"""Tests for the Gumbel-VQ model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import GumbelQuantizedAutoencoderConfig, GumbelQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class GumbelQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = GumbelQuantizedAutoencoderConfig(input_dim=16, latent_dim=4, hidden_dims=[12, 8], codebook_size=32)

    def test_forward_returns_assignment_metrics(self) -> None:
        model = GumbelQuantizedAutoencoderModel(self.config)
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4,))
        self.assertIn("assignment_entropy", outputs.loss_dict)

    def test_export_includes_codebook(self) -> None:
        model = GumbelQuantizedAutoencoderModel(self.config)
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "gumbel_quantized_autoencoder")
        self.assertEqual(tuple(artifact.extras["codebook"].shape), (32, 4))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = GumbelQuantizedAutoencoderModel(self.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = GumbelQuantizedAutoencoderModel.from_pretrained(tmpdir)
        self.assertTrue(loaded.config.straight_through)


if __name__ == "__main__":
    unittest.main()
