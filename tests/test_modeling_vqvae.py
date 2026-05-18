"""Tests for the vector-quantized autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency gate
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import VectorQuantizedAutoencoderConfig, VectorQuantizedAutoencoderModel
    from autoencoders.data import TensorSpec


@unittest.skipIf(torch is None, "torch is required for model tests")
class VectorQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 5, 16)
        self.config = VectorQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=32,
            commitment_weight=0.25,
            codebook_weight=1.0,
        )

    def test_forward_returns_quantized_fields(self) -> None:
        model = VectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 5, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 5, 4))
        self.assertEqual(tuple(outputs.quantized_latents.shape), (4, 5, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 5))
        self.assertIn("commitment_loss", outputs.loss_dict)
        self.assertIn("codebook_loss", outputs.loss_dict)
        expected_loss = (
            outputs.reconstruction_loss
            + self.config.commitment_weight * outputs.commitment_loss
            + self.config.codebook_weight * outputs.codebook_loss
        )
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))

    def test_ema_codebook_disables_codebook_loss_term(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=32,
            commitment_weight=0.25,
            codebook_weight=1.0,
            use_ema_codebook=True,
        )
        model = VectorQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()
        initial_codebook = model.codebook.weight.detach().clone()

        outputs = model(inputs=self.inputs)

        self.assertTrue(torch.allclose(outputs.codebook_loss, torch.zeros_like(outputs.codebook_loss)))
        expected_loss = outputs.reconstruction_loss + config.commitment_weight * outputs.commitment_loss
        self.assertTrue(torch.allclose(outputs.loss, expected_loss))
        self.assertFalse(model.codebook.weight.requires_grad)
        self.assertFalse(torch.equal(initial_codebook, model.codebook.weight.detach()))

    def test_export_includes_codebook_artifacts(self) -> None:
        model = VectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        artifact = model.export(self.inputs, metadata={"split": "test"})

        self.assertEqual(artifact.model_type, "vector_quantized_autoencoder")
        self.assertEqual(tuple(artifact.latents.shape), (4, 5, 4))
        self.assertEqual(tuple(artifact.encoded.shape), (4, 5, 8))
        self.assertEqual(tuple(artifact.reconstruction.shape), (4, 5, 16))
        self.assertEqual(tuple(artifact.quantized_latents.shape), (4, 5, 4))
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4, 5))
        self.assertEqual(artifact.metadata["input_shape"], [4, 5, 16])
        self.assertEqual(artifact.metadata["latent_shape"], [4, 5, 4])
        self.assertEqual(artifact.metadata["split"], "test")
        self.assertEqual(artifact.extras["codebook_size"], 32)
        self.assertEqual(tuple(artifact.extras["codebook"].shape), (32, 4))
        self.assertIsNot(artifact.extras["codebook"], model.codebook.weight)
        self.assertTrue(torch.equal(artifact.extras["codebook"], model.codebook.weight.detach()))

    def test_export_can_skip_reconstruction(self) -> None:
        model = VectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))

        artifact = model.export(self.inputs, include_reconstruction=False)

        self.assertIsNone(artifact.reconstruction)
        self.assertEqual(tuple(artifact.quantized_latents.shape), (4, 5, 4))
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4, 5))

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = VectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.05)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = VectorQuantizedAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.codebook_size, 32)
        self.assertEqual(loaded.config.commitment_weight, 0.25)
        self.assertEqual(loaded.config.codebook_weight, 1.0)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)

    def test_reset_dead_codes_reinitializes_selected_embeddings(self) -> None:
        model = VectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config))
        with torch.no_grad():
            original_codebook = model.codebook.weight.detach().clone()
            reference_latents = torch.randn(6, 4)
            dead_code_mask = torch.tensor([False, True, False, True] + [False] * 28)

            reset_count = model.reset_dead_codes(dead_code_mask, reference_latents)

        self.assertEqual(reset_count, 2)
        self.assertTrue(torch.equal(model.codebook.weight[~dead_code_mask], original_codebook[~dead_code_mask]))
        self.assertFalse(torch.equal(model.codebook.weight[dead_code_mask], original_codebook[dead_code_mask]))

    def test_dead_code_reset_requires_last_step_signal_when_enabled(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=32,
            dead_code_reset=True,
        )
        model = VectorQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()

        with self.assertRaisesRegex(ValueError, "is_last_train_step"):
            model(inputs=self.inputs)

    def test_vector_vq_requires_multi_vector_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank >= 2"):
            VectorQuantizedAutoencoderModel(
                config=self.config,
                sample_spec=TensorSpec(shape=(16,)),
                encoder="mlp",
                decoder="mlp",
                encoder_config={"hidden_dims": [12, 8], "activation": "relu", "use_bias": True},
                decoder_config={"hidden_dims": [12, 16], "activation": "relu", "use_bias": True},
            )

    def test_dead_code_reset_count_is_recorded_on_last_train_step(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=32,
            dead_code_reset=True,
            dead_code_threshold=100,
        )
        model = VectorQuantizedAutoencoderModel(config=config, **build_mlp_backbone_kwargs_from_model_config(config))
        model.train()

        _ = model(inputs=self.inputs, is_last_train_step=True)

        self.assertGreater(model.consume_dead_code_reset_count(), 0)
        self.assertEqual(model.consume_dead_code_reset_count(), 0)

    def test_forward_supports_cnn_backbone_on_image_inputs(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            latent_dim=64,
            codebook_size=32,
            commitment_weight=0.25,
            codebook_weight=1.0,
        )
        model = VectorQuantizedAutoencoderModel(
            config=config,
            sample_spec=TensorSpec(shape=(32, 32, 3)),
            encoder="cnn",
            encoder_config={
                "channels": [64, 128],
                "kernel_sizes": [4, 4],
                "strides": [2, 2],
                "paddings": [1, 1],
                "activation": "relu",
                "use_bias": True,
            },
        )
        image_inputs = torch.rand(2, 32, 32, 3)

        outputs = model(inputs=image_inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (2, 32, 32, 3))
        self.assertEqual(tuple(outputs.encoded.shape), (2, 8, 8, 128))
        self.assertEqual(tuple(outputs.latents.shape), (2, 8, 8, 64))
        self.assertEqual(tuple(outputs.quantized_latents.shape), (2, 8, 8, 64))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (2, 8, 8))

    def test_forward_supports_vision_transformer_backbone_on_image_inputs(self) -> None:
        config = VectorQuantizedAutoencoderConfig(
            latent_dim=64,
            codebook_size=32,
            commitment_weight=0.25,
            codebook_weight=1.0,
        )
        model = VectorQuantizedAutoencoderModel(
            config=config,
            sample_spec=TensorSpec(shape=(32, 32, 3)),
            encoder="vision_transformer",
            encoder_config={
                "patch_size": 4,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 8,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "use_bias": True,
            },
        )
        image_inputs = torch.rand(2, 32, 32, 3)

        outputs = model(inputs=image_inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (2, 32, 32, 3))
        self.assertEqual(tuple(outputs.encoded.shape), (2, 64, 128))
        self.assertEqual(tuple(outputs.latents.shape), (2, 64, 64))
        self.assertEqual(tuple(outputs.quantized_latents.shape), (2, 64, 64))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (2, 64))


if __name__ == "__main__":
    unittest.main()
