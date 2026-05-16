"""Tests for the denoising variational autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import build_mlp_backbone_kwargs_from_model_config, DenoisingVariationalAutoencoderConfig, DenoisingVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class DenoisingVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = DenoisingVariationalAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            kl_weight=0.5,
            noise_type="gaussian",
            noise_std=0.1,
        )

    def test_forward_returns_expected_fields(self) -> None:
        model = DenoisingVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.train()
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertIn("kl_loss", outputs.loss_dict)
        self.assertIn("corrupted_inputs", outputs.hidden_states)

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = DenoisingVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = DenoisingVariationalAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.noise_type, "gaussian")


if __name__ == "__main__":
    unittest.main()
