"""Tests for the information variational autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from autoencoders import InformationVariationalAutoencoderConfig, InformationVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class InformationVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = InformationVariationalAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            kl_weight=0.2,
            mmd_weight=3.0,
        )

    def test_forward_returns_expected_fields(self) -> None:
        model = InformationVariationalAutoencoderModel(self.config)
        model.train()

        outputs = model(inputs=self.inputs, current_epoch=1)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertIn("reconstruction_loss", outputs.loss_dict)
        self.assertIn("kl_loss", outputs.loss_dict)
        self.assertIn("free_bits_kl_loss", outputs.loss_dict)
        self.assertIn("mmd_loss", outputs.loss_dict)
        expected_loss = (
            outputs.reconstruction_loss
            + outputs.effective_kl_weight * outputs.free_bits_kl_loss
            + self.config.mmd_weight * outputs.mmd_loss
        )
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_return_dict_false_returns_tuple(self) -> None:
        model = InformationVariationalAutoencoderModel(self.config)
        outputs = model(inputs=self.inputs, current_epoch=1, return_dict=False)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[1].shape), (4, 16))
        self.assertEqual(tuple(outputs[2].shape), (4, 4))

    def test_export_includes_posterior_statistics(self) -> None:
        model = InformationVariationalAutoencoderModel(self.config)
        model.eval()

        artifact = model.export(self.inputs)

        self.assertEqual(artifact.model_type, "information_variational_autoencoder")
        self.assertEqual(tuple(artifact.posterior_mean.shape), (4, 4))
        self.assertEqual(tuple(artifact.posterior_logvar.shape), (4, 4))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = InformationVariationalAutoencoderModel(self.config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = InformationVariationalAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.mmd_weight, 3.0)
        self.assertEqual(loaded.config.mmd_bandwidths, [0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
