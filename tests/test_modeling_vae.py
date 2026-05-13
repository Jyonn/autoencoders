"""Tests for the variational autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from autoencoders import VariationalAutoencoderConfig, VariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class VariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = VariationalAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            kl_weight=0.5,
        )

    def test_forward_returns_expected_fields(self) -> None:
        model = VariationalAutoencoderModel(self.config)
        model.train()

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.posterior_mean.shape), (4, 4))
        self.assertEqual(tuple(outputs.posterior_logvar.shape), (4, 4))
        self.assertIn("reconstruction_loss", outputs.loss_dict)
        self.assertIn("kl_loss", outputs.loss_dict)
        self.assertIn("loss", outputs.loss_dict)
        expected_loss = outputs.reconstruction_loss + self.config.kl_weight * outputs.kl_loss
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_eval_uses_mean_by_default(self) -> None:
        model = VariationalAutoencoderModel(self.config)
        model.eval()

        outputs = model(inputs=self.inputs)

        self.assertTrue(torch.allclose(outputs.latents, outputs.posterior_mean))

    def test_sample_posterior_override_changes_latents(self) -> None:
        model = VariationalAutoencoderModel(self.config)
        model.eval()

        outputs = model(inputs=self.inputs, sample_posterior=True)

        self.assertFalse(torch.allclose(outputs.latents, outputs.posterior_mean))

    def test_return_dict_false_returns_tuple(self) -> None:
        model = VariationalAutoencoderModel(self.config)

        outputs = model(inputs=self.inputs, return_dict=False)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[1].shape), (4, 16))
        self.assertEqual(tuple(outputs[2].shape), (4, 4))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = VariationalAutoencoderModel(self.config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = VariationalAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.kl_weight, 0.5)
        self.assertTrue(loaded.config.use_mean_in_eval)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
