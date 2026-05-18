"""Tests for the VQ-VAE-2 model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import HierarchicalVectorQuantizedAutoencoderConfig, HierarchicalVectorQuantizedAutoencoderModel
    from autoencoders.data import TensorSpec
    from autoencoders.modules import CNNModuleConfig


@unittest.skipIf(torch is None, "torch is required for model tests")
class HierarchicalVectorQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 5, 16)
        self.config = HierarchicalVectorQuantizedAutoencoderConfig(
            latent_dim=4,
            top_latent_dim=3,
            hidden_dims=[12, 8],
            codebook_size=16,
        )

    def test_forward_returns_hierarchical_quantization_fields(self) -> None:
        model = HierarchicalVectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        outputs = model(inputs=self.inputs)
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 5, 2))
        self.assertEqual(tuple(outputs.top_quantized_latents.shape), (4, 5, 3))
        self.assertEqual(tuple(outputs.bottom_quantized_latents.shape), (4, 5, 4))

    def test_export_includes_both_codebooks(self) -> None:
        model = HierarchicalVectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        artifact = model.export(self.inputs)
        self.assertEqual(artifact.model_type, "hierarchical_vector_quantized_autoencoder")
        self.assertEqual(tuple(artifact.extras["top_codebook"].shape), (16, 3))
        self.assertEqual(tuple(artifact.extras["bottom_codebook"].shape), (16, 4))

    def test_vqvae2_requires_multi_vector_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank >= 2"):
            HierarchicalVectorQuantizedAutoencoderModel(
                config=self.config,
                sample_spec=TensorSpec(shape=(16,)),
                encoder="mlp",
                decoder="mlp",
                encoder_config={"hidden_dims": [12, 8], "activation": "relu", "use_bias": True},
                decoder_config={"hidden_dims": [12, 16], "activation": "relu", "use_bias": True},
            )

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = HierarchicalVectorQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = HierarchicalVectorQuantizedAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.top_latent_dim, 3)

    def test_auto_decoder_from_cnn_encoder_is_rejected_for_hierarchical_latents(self) -> None:
        image_config = HierarchicalVectorQuantizedAutoencoderConfig(
            latent_dim=64,
            top_latent_dim=32,
            codebook_size=16,
            commitment_weight=0.25,
            reconstruction_loss="mse",
        )
        with self.assertRaisesRegex(ValueError, "Provide an explicit decoder"):
            HierarchicalVectorQuantizedAutoencoderModel(
                config=image_config,
                sample_spec=TensorSpec(shape=(32, 32, 3)),
                encoder="cnn",
                decoder=None,
                encoder_config=CNNModuleConfig(
                    channels=[64, 128],
                    kernel_sizes=[4, 4],
                    strides=[2, 2],
                    paddings=[1, 1],
                    activation="relu",
                    use_bias=True,
                ),
            )

    def test_kmeans_init_initializes_top_and_bottom_codebooks(self) -> None:
        config = HierarchicalVectorQuantizedAutoencoderConfig(
            latent_dim=4,
            top_latent_dim=3,
            hidden_dims=[12, 8],
            codebook_size=8,
            kmeans_init=True,
            kmeans_iters=3,
        )
        model = HierarchicalVectorQuantizedAutoencoderModel(
            config=config,
            **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=16),
        )

        self.assertFalse(model.codebooks_initialized)
        self.assertTrue(torch.equal(model.top_codebook.weight, torch.zeros_like(model.top_codebook.weight)))
        self.assertTrue(torch.equal(model.bottom_codebook.weight, torch.zeros_like(model.bottom_codebook.weight)))

        _ = model(inputs=self.inputs)

        self.assertTrue(model.codebooks_initialized)
        self.assertFalse(torch.equal(model.top_codebook.weight, torch.zeros_like(model.top_codebook.weight)))
        self.assertFalse(torch.equal(model.bottom_codebook.weight, torch.zeros_like(model.bottom_codebook.weight)))

    def test_sinkhorn_assignment_runs_for_hierarchical_codebooks(self) -> None:
        config = HierarchicalVectorQuantizedAutoencoderConfig(
            latent_dim=4,
            top_latent_dim=3,
            hidden_dims=[12, 8],
            codebook_size=8,
            assignment_strategy="sinkhorn",
            sinkhorn_epsilon=[0.0, 0.01],
            sinkhorn_iters=10,
        )
        model = HierarchicalVectorQuantizedAutoencoderModel(
            config=config,
            **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=16),
        )

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.top_quantized_latents.shape), (4, 5, 3))
        self.assertEqual(tuple(outputs.bottom_quantized_latents.shape), (4, 5, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 5, 2))

    def test_sinkhorn_assignment_validates_hierarchical_epsilon_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 values"):
            HierarchicalVectorQuantizedAutoencoderConfig(
                latent_dim=4,
                top_latent_dim=3,
                hidden_dims=[12, 8],
                codebook_size=8,
                assignment_strategy="sinkhorn",
                sinkhorn_epsilon=[0.01, 0.02, 0.03],
            )


if __name__ == "__main__":
    unittest.main()
