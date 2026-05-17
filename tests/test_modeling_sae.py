"""Tests for the sparse autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import SparseAutoencoderConfig, SparseAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class SparseAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = SparseAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            sparsity_weight=0.05,
        )

    def test_forward_returns_sparse_loss(self) -> None:
        model = SparseAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertIn("reconstruction_loss", outputs.loss_dict)
        self.assertIn("sparsity_loss", outputs.loss_dict)
        expected_loss = outputs.reconstruction_loss + self.config.sparsity_weight * outputs.sparsity_loss
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = SparseAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.15)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SparseAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.sparsity_weight, 0.05)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
