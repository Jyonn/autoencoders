"""Tests for the vector-quantized autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from autoencoders import VectorQuantizedAutoencoderConfig, VectorQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class VectorQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = VectorQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=32,
            commitment_weight=0.25,
            codebook_weight=1.0,
        )

    def test_forward_returns_quantized_fields(self) -> None:
        model = VectorQuantizedAutoencoderModel(self.config)

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4,))
        self.assertIn("commitment_loss", outputs.loss_dict)
        self.assertIn("codebook_loss", outputs.loss_dict)
        expected_loss = (
            outputs.reconstruction_loss
            + self.config.commitment_weight * outputs.commitment_loss
            + self.config.codebook_weight * outputs.codebook_loss
        )
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_export_includes_codebook_artifacts(self) -> None:
        model = VectorQuantizedAutoencoderModel(self.config)
        artifact = model.export(self.inputs, metadata={"split": "test"})

        self.assertEqual(artifact.model_type, "vector_quantized_autoencoder")
        self.assertEqual(tuple(artifact.latents.shape), (4, 4))
        self.assertEqual(tuple(artifact.encoded.shape), (4, 4))
        self.assertEqual(tuple(artifact.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(artifact.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4,))
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4,))
        self.assertEqual(artifact.metadata["input_shape"], [4, 16])
        self.assertEqual(artifact.metadata["latent_shape"], [4, 4])
        self.assertEqual(artifact.metadata["split"], "test")
        self.assertEqual(artifact.extras["codebook_size"], 32)
        self.assertEqual(tuple(artifact.extras["codebook"].shape), (32, 4))
        self.assertIsNot(artifact.extras["codebook"], model.codebook.weight)
        self.assertTrue(torch.equal(artifact.extras["codebook"], model.codebook.weight.detach()))

    def test_export_can_skip_reconstruction(self) -> None:
        model = VectorQuantizedAutoencoderModel(self.config)

        artifact = model.export(self.inputs, include_reconstruction=False)

        self.assertIsNone(artifact.reconstruction)
        self.assertEqual(tuple(artifact.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4,))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = VectorQuantizedAutoencoderModel(self.config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.05)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = VectorQuantizedAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.codebook_size, 32)
        self.assertEqual(loaded.config.commitment_weight, 0.25)
        self.assertEqual(loaded.config.codebook_weight, 1.0)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
