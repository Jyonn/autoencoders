"""Tests for the basic autoencoder model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from torch import nn

    from autoencoders import AutoencoderConfig, AutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class AutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            activation="relu",
        )
        self.inputs = torch.randn(3, 16)

    def test_forward_returns_expected_shapes(self) -> None:
        model = AutoencoderModel(self.config)

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (3, 16))
        self.assertEqual(tuple(outputs.latents.shape), (3, 4))
        self.assertEqual(tuple(outputs.encoded.shape), (3, 4))
        self.assertIn("reconstruction_loss", outputs.loss_dict)

    def test_forward_uses_inputs_argument(self) -> None:
        model = AutoencoderModel(self.config)

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (3, 16))
        self.assertEqual(tuple(outputs.latents.shape), (3, 4))

    def test_return_dict_false_returns_tuple(self) -> None:
        model = AutoencoderModel(self.config)

        outputs = model(inputs=self.inputs, return_dict=False)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[1].shape), (3, 16))
        self.assertEqual(tuple(outputs[2].shape), (3, 4))

    def test_reconstruct_matches_output_shape(self) -> None:
        model = AutoencoderModel(self.config)

        reconstruction = model.reconstruct(self.inputs)

        self.assertEqual(tuple(reconstruction.shape), (3, 16))

    def test_export_returns_standard_artifact(self) -> None:
        model = AutoencoderModel(self.config)

        artifact = model.export(self.inputs, metadata={"split": "test"})

        self.assertEqual(artifact.model_type, "autoencoder")
        self.assertEqual(tuple(artifact.latents.shape), (3, 4))
        self.assertEqual(tuple(artifact.reconstruction.shape), (3, 16))
        self.assertEqual(artifact.metadata["input_shape"], [3, 16])
        self.assertEqual(artifact.metadata["latent_shape"], [3, 4])
        self.assertEqual(artifact.metadata["split"], "test")

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = AutoencoderModel(self.config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = AutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.input_dim, 16)
        self.assertEqual(loaded.config.latent_dim, 4)

        original_state = model.state_dict()
        loaded_state = loaded.state_dict()
        for name, parameter in original_state.items():
            self.assertTrue(torch.equal(parameter, loaded_state[name]), msg=name)

    def test_save_pretrained_writes_builtin_module_specs(self) -> None:
        model = AutoencoderModel(self.config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            self.assertTrue((Path(tmpdir) / "encoder_module.json").exists())
            self.assertTrue((Path(tmpdir) / "decoder_module.json").exists())

    def test_external_modules_require_reinjection_on_load(self) -> None:
        encoder = nn.Linear(16, 4)
        decoder = nn.Linear(4, 16)
        model = AutoencoderModel(self.config, encoder=encoder, decoder=decoder)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            with self.assertRaisesRegex(ValueError, "external encoder module"):
                AutoencoderModel.from_pretrained(tmpdir)
            loaded = AutoencoderModel.from_pretrained(tmpdir, encoder=nn.Linear(16, 4), decoder=nn.Linear(4, 16))

        self.assertIsInstance(loaded.encoder, nn.Linear)


if __name__ == "__main__":
    unittest.main()
