"""Tests for the residual-quantized autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from tests._mlp_helpers import build_mlp_backbone_kwargs_from_model_config
    from autoencoders import ResidualQuantizedAutoencoderConfig, ResidualQuantizedAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class ResidualQuantizedAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = ResidualQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=16,
            num_quantizers=3,
        )

    def test_forward_returns_residual_quantizer_indices(self) -> None:
        model = ResidualQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.reconstruction.shape), (4, 16))
        self.assertEqual(tuple(outputs.latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 3))
        self.assertIn("commitment_loss", outputs.loss_dict)
        self.assertIn("codebook_loss", outputs.loss_dict)

    def test_export_includes_residual_codebooks(self) -> None:
        model = ResidualQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))

        artifact = model.export(self.inputs, metadata={"split": "test"})

        self.assertEqual(artifact.model_type, "residual_quantized_autoencoder")
        self.assertEqual(tuple(artifact.codebook_indices.shape), (4, 3))
        self.assertEqual(artifact.extras["num_quantizers"], 3)
        self.assertEqual(tuple(artifact.extras["codebooks"].shape), (3, 16, 4))
        self.assertEqual(artifact.metadata["split"], "test")

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = ResidualQuantizedAutoencoderModel(config=self.config, **build_mlp_backbone_kwargs_from_model_config(self.config, feature_dim=16))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.04)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = ResidualQuantizedAutoencoderModel.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.num_quantizers, 3)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]), msg=name)

    def test_kmeans_init_initializes_all_residual_codebooks(self) -> None:
        config = ResidualQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=8,
            num_quantizers=3,
            kmeans_init=True,
            kmeans_iters=3,
        )
        model = ResidualQuantizedAutoencoderModel(
            config=config,
            **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=16),
        )

        self.assertFalse(model.codebooks_initialized)
        for codebook in model.codebooks:
            self.assertTrue(torch.equal(codebook.weight, torch.zeros_like(codebook.weight)))

        _ = model(inputs=self.inputs)

        self.assertTrue(model.codebooks_initialized)
        self.assertFalse(torch.equal(model.codebooks[0].weight, torch.zeros_like(model.codebooks[0].weight)))
        self.assertFalse(torch.equal(model.ema_weight_sum[0], torch.zeros_like(model.ema_weight_sum[0])))

    def test_sinkhorn_assignment_runs_for_residual_quantizers(self) -> None:
        config = ResidualQuantizedAutoencoderConfig(
            latent_dim=4,
            hidden_dims=[12, 8],
            codebook_size=8,
            num_quantizers=3,
            assignment_strategy="sinkhorn",
            sinkhorn_epsilon=[0.0, 0.01, 0.02],
            sinkhorn_iters=10,
        )
        model = ResidualQuantizedAutoencoderModel(
            config=config,
            **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=16),
        )

        outputs = model(inputs=self.inputs)

        self.assertEqual(tuple(outputs.quantized_latents.shape), (4, 4))
        self.assertEqual(tuple(outputs.codebook_indices.shape), (4, 3))

    def test_sinkhorn_assignment_validates_per_quantizer_epsilon_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "3 values"):
            ResidualQuantizedAutoencoderConfig(
                latent_dim=4,
                hidden_dims=[12, 8],
                codebook_size=8,
                num_quantizers=3,
                assignment_strategy="sinkhorn",
                sinkhorn_epsilon=[0.01, 0.02],
            )

    def test_rqvae_ema_updates_use_forward_residual_chain(self) -> None:
        config = ResidualQuantizedAutoencoderConfig(
            latent_dim=1,
            hidden_dims=[2],
            codebook_size=1,
            num_quantizers=2,
            use_ema_codebook=True,
            ema_decay=0.0,
            ema_epsilon=1e-5,
        )
        model = ResidualQuantizedAutoencoderModel(
            config=config,
            **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=1),
        )
        model.codebooks[0].weight.data.fill_(1.0)
        model.codebooks[1].weight.data.fill_(2.0)
        model.ema_cluster_size.fill_(1.0)
        model.ema_weight_sum.zero_()

        encoded = torch.tensor([[10.0]])
        indices = torch.tensor([[0, 0]], dtype=torch.long)

        model._update_ema_codebooks(encoded, indices)

        self.assertAlmostEqual(float(model.codebooks[0].weight.item()), 10.0, places=6)
        self.assertAlmostEqual(float(model.codebooks[1].weight.item()), 9.0, places=6)

    def test_rqvae_uses_mean_per_quantizer_losses(self) -> None:
        config = ResidualQuantizedAutoencoderConfig(
            latent_dim=1,
            hidden_dims=[2],
            codebook_size=1,
            num_quantizers=2,
        )
        model = ResidualQuantizedAutoencoderModel(
            config=config,
            **build_mlp_backbone_kwargs_from_model_config(config, feature_dim=1),
        )
        encoded = torch.tensor([[10.0]])
        model.codebooks[0].weight.data.fill_(1.0)
        model.codebooks[1].weight.data.fill_(2.0)

        quantized, indices = model.quantize(encoded)

        self.assertEqual(tuple(quantized.shape), (1, 1))
        self.assertEqual(tuple(indices.shape), (1, 2))
        self.assertAlmostEqual(float(model.compute_commitment_loss(encoded, quantized).detach()), 65.0, places=6)
        self.assertAlmostEqual(float(model.compute_codebook_loss(encoded, quantized).detach()), 65.0, places=6)


if __name__ == "__main__":
    unittest.main()
