"""Tests for the denoising autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import DenoisingAutoencoderConfig, DenoisingAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class DenoisingAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)

    def test_forward_uses_noisy_inputs_during_training(self) -> None:
        config = DenoisingAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            noise_type="gaussian",
            noise_std=0.5,
        )
        model = DenoisingAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertFalse(torch.equal(outputs.hidden_states["corrupted_inputs"], self.inputs))
        self.assertTrue(torch.equal(outputs.hidden_states["inputs"], self.inputs))

    def test_eval_skips_noise_by_default(self) -> None:
        config = DenoisingAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            noise_type="gaussian",
            noise_std=0.5,
        )
        model = DenoisingAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.eval()

        outputs = model(inputs=self.inputs)

        self.assertTrue(torch.equal(outputs.hidden_states["corrupted_inputs"], self.inputs))

    def test_masking_noise_zeros_some_entries(self) -> None:
        config = DenoisingAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            noise_type="masking",
            masking_ratio=0.5,
        )
        model = DenoisingAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()

        outputs = model(inputs=torch.ones(4, 16))

        corrupted_inputs = outputs.hidden_states["corrupted_inputs"]
        self.assertTrue(torch.any(corrupted_inputs == 0))
        self.assertTrue(torch.all((corrupted_inputs == 0) | (corrupted_inputs == 1)))

    def test_explicit_corrupted_inputs_override_noise(self) -> None:
        config = DenoisingAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            noise_type="gaussian",
            noise_std=0.5,
        )
        model = DenoisingAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()
        corrupted = torch.zeros_like(self.inputs)

        outputs = model(inputs=self.inputs, corrupted_inputs=corrupted)

        self.assertTrue(torch.equal(outputs.hidden_states["corrupted_inputs"], corrupted))
        self.assertTrue(torch.equal(outputs.hidden_states["inputs"], self.inputs))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        config = DenoisingAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            noise_type="masking",
            masking_ratio=0.25,
        )
        model = DenoisingAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.125)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = DenoisingAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.noise_type, "masking")
        self.assertEqual(loaded.config.masking_ratio, 0.25)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
