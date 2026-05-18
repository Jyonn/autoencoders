"""Tests for the VampPrior variational autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import VampPriorVariationalAutoencoderConfig, VampPriorVariationalAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class VampPriorVariationalAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = VampPriorVariationalAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[12, 8],
            kl_weight=0.2,
            num_pseudo_inputs=6,
        )

    def test_forward_returns_expected_fields(self) -> None:
        model = VampPriorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        model.train()

        outputs = model(inputs=self.inputs, current_epoch=1)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.posterior_mean.shape), (4, 4))
        self.assertEqual(tuple(outputs.pseudo_posterior_mean.shape), (6, 4))
        self.assertIn("kl_loss", outputs.loss_dict)
        self.assertIn("free_bits_kl_loss", outputs.loss_dict)

    def test_export_includes_posterior_statistics(self) -> None:
        model = VampPriorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        model.eval()

        artifact = model.export(self.inputs)

        self.assertEqual(artifact.model_type, "vamp_prior_variational_autoencoder")
        self.assertEqual(tuple(artifact.posterior_mean.shape), (4, 4))
        self.assertEqual(tuple(artifact.posterior_logvar.shape), (4, 4))

    def test_return_dict_false_returns_tuple(self) -> None:
        model = VampPriorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        outputs = model(inputs=self.inputs, current_epoch=1, return_dict=False)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[1].shape), (4, 16))
        self.assertEqual(tuple(outputs[2].shape), (4, 4))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = VampPriorVariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = VampPriorVariationalAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.num_pseudo_inputs, 6)
        self.assertEqual(loaded.config.pseudo_input_std, 0.01)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
