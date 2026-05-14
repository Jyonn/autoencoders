"""Tests for the VQ-VAE-2 model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import HierarchicalVectorQuantizedAutoencoderConfig, HierarchicalVectorQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class HierarchicalVectorQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 5, 16)
        self.config = HierarchicalVectorQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            top_latent_dim=3,
            hidden_dims=[12, 8],
            codebook_size=16,
        )

    def test_forward_returns_hierarchical_quantization_fields(self) -> None:
        model = HierarchicalVectorQuantizedAutoencoderModel(self.config)
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 5, 2))
        self.assertEqual(tuple(outputs.top_quantized_latents.shape), (4, 5, 3))
        self.assertEqual(tuple(outputs.bottom_quantized_latents.shape), (4, 5, 4))

    def test_export_includes_both_codebooks(self) -> None:
        model = HierarchicalVectorQuantizedAutoencoderModel(self.config)
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "hierarchical_vector_quantized_autoencoder")
        self.assertEqual(tuple(artifact.extras["top_codebook"].shape), (16, 3))
        self.assertEqual(tuple(artifact.extras["bottom_codebook"].shape), (16, 4))

    def test_vqvae2_requires_multi_vector_inputs(self) -> None:
        model = HierarchicalVectorQuantizedAutoencoderModel(self.config)
        with self.assertRaisesRegex(ValueError, "rank >= 3"):
            model(inputs=torch.randn(4, 16))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = HierarchicalVectorQuantizedAutoencoderModel(self.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = HierarchicalVectorQuantizedAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.top_latent_dim, 3)


if __name__ == "__main__":
    unittest.main()
