"""Tests for the residual-quantized autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import ResidualQuantizedAutoencoderConfig, ResidualQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class ResidualQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = ResidualQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=16,
            num_quantizers=3,
        )

    def test_forward_returns_residual_quantizer_indices(self) -> None:
        model = ResidualQuantizedAutoencoderModel(self.config)

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 3))
        self.assertIn("commitment_loss", outputs.loss_dict)
        self.assertIn("codebook_loss", outputs.loss_dict)

    def test_export_includes_residual_codebooks(self) -> None:
        model = ResidualQuantizedAutoencoderModel(self.config)

        artifact = model.export(self.inputs, metadata={"split": "test"})

        self.assertEqual(artifact.model_type, "residual_quantized_autoencoder")
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4, 3))
        self.assertEqual(artifact.extras["num_quantizers"], 3)
        self.assertEqual(tuple(artifact.extras["codebooks"].shape), (3, 16, 4))
        self.assertEqual(artifact.metadata["split"], "test")

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = ResidualQuantizedAutoencoderModel(self.config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.04)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = ResidualQuantizedAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.num_quantizers, 3)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
