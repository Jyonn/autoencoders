"""Tests for the product-quantized autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import build_mlp_backbone_kwargs_from_model_config, ProductQuantizedAutoencoderConfig, ProductQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class ProductQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = ProductQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=16,
            num_codebooks=2,
        )

    def test_forward_returns_multi_codebook_indices(self) -> None:
        model = ProductQuantizedAutoencoderModel(self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 2))
        self.assertIn("commitment_loss", outputs.loss_dict)
        self.assertIn("codebook_loss", outputs.loss_dict)

    def test_export_includes_product_codebooks(self) -> None:
        model = ProductQuantizedAutoencoderModel(self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))

        artifact = model.export(self.inputs, metadata={"split": "test"})

        self.assertEqual(artifact.model_type, "product_quantized_autoencoder")
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4, 2))
        self.assertEqual(artifact.extras["num_codebooks"], 2)
        self.assertEqual(tuple(artifact.extras["codebooks"].shape), (2, 16, 2))
        self.assertEqual(artifact.metadata["split"], "test")

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = ProductQuantizedAutoencoderModel(self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.03)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = ProductQuantizedAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.num_codebooks, 2)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
