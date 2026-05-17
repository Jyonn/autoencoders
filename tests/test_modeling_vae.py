"""Tests for the variational autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from pathlib import Path
    from torch import nn

    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
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
        model = VariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.train()

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.posterior_mean.shape), (4, 4))
        self.assertEqual(tuple(outputs.posterior_logvar.shape), (4, 4))
        self.assertIn("reconstruction_loss", outputs.loss_dict)
        self.assertIn("kl_loss", outputs.loss_dict)
        self.assertIn("free_bits_kl_loss", outputs.loss_dict)
        self.assertIn("effective_kl_weight", outputs.loss_dict)
        self.assertIn("loss", outputs.loss_dict)
        expected_loss = outputs.reconstruction_loss + self.config.kl_weight * outputs.free_bits_kl_loss
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_training_forward_requires_current_epoch_when_warmup_is_enabled(self) -> None:
        config = VariationalAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            kl_weight=0.5,
            kl_warmup_epochs=3,
            kl_start_weight=0.0,
        )
        model = VariationalAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()

        with self.assertRaisesRegex(ValueError, "current_epoch"):
            model(inputs=self.inputs)

    def test_decoder_is_inferred_from_builtin_encoder_when_omitted(self) -> None:
        model = VariationalAutoencoderModel(
            config=self.config,
            encoder="mlp",
            encoder_config={"hidden_dims": [12, 8], "activation": "relu", "use_bias": True},
        )
        model.eval()

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))

    def test_training_forward_uses_warmup_weight_when_current_epoch_is_provided(self) -> None:
        config = VariationalAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            kl_weight=0.6,
            kl_warmup_epochs=3,
            kl_start_weight=0.0,
        )
        model = VariationalAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()

        outputs = model(inputs=self.inputs, current_epoch=2)

        self.assertAlmostEqual(float(outputs.effective_kl_weight), 0.3, places=6)
        expected_loss = outputs.reconstruction_loss + outputs.effective_kl_weight * outputs.free_bits_kl_loss
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_eval_uses_mean_by_default(self) -> None:
        model = VariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.eval()

        outputs = model(inputs=self.inputs)

        self.assertTrue(torch.allclose(outputs.latents, outputs.posterior_mean))

    def test_sample_posterior_override_changes_latents(self) -> None:
        model = VariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.eval()

        outputs = model(inputs=self.inputs, sample_posterior=True)

        self.assertFalse(torch.allclose(outputs.latents, outputs.posterior_mean))

    def test_return_dict_false_returns_tuple(self) -> None:
        model = VariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))

        outputs = model(inputs=self.inputs, return_dict=False)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[1].shape), (4, 16))
        self.assertEqual(tuple(outputs[2].shape), (4, 4))

    def test_export_includes_posterior_statistics(self) -> None:
        model = VariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        model.eval()

        artifact = model.export(self.inputs)

        self.assertEqual(artifact.model_type, "variational_autoencoder")
        self.assertEqual(tuple(artifact.latents.shape), (4, 4))
        self.assertEqual(tuple(artifact.posterior_mean.shape), (4, 4))
        self.assertEqual(tuple(artifact.posterior_logvar.shape), (4, 4))
        self.assertTrue(torch.allclose(artifact.latents, artifact.posterior_mean))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = VariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
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

    def test_save_pretrained_writes_builtin_module_specs(self) -> None:
        model = VariationalAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            self.assertTrue((Path(tmpdir) / "encoder_module.json").exists())
            self.assertTrue((Path(tmpdir) / "decoder_module.json").exists())

    def test_external_modules_require_reinjection_on_load(self) -> None:
        encoder = nn.Linear(16, 8)
        decoder = nn.Linear(4, 16)
        model = VariationalAutoencoderModel(config=self.config, encoder=encoder, decoder=decoder)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            with self.assertRaisesRegex(ValueError, "external encoder module"):
                VariationalAutoencoderModel.from_pretrained(tmpdir)
            loaded = VariationalAutoencoderModel.from_pretrained(
                tmpdir,
                encoder=nn.Linear(16, 8),
                decoder=nn.Linear(4, 16),
            )

        self.assertIsInstance(loaded.encoder, nn.Linear)

    def test_decoder_none_with_non_backbone_encoder_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot infer decoder"):
            VariationalAutoencoderModel(config=self.config, encoder=nn.Linear(16, 8))


if __name__ == "__main__":
    unittest.main()
