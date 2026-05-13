"""Tests for the top-k sparse autoencoder model."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from autoencoders import TopKSparseAutoencoderConfig, TopKSparseAutoencoderModel


@unittest.skipIf(torch is None, "torch is required for model tests")
class TopKSparseAutoencoderModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = torch.randn(4, 16)
        self.config = TopKSparseAutoencoderConfig(
            input_dim=16,
            latent_dim=6,
            hidden_dims=[12, 8],
            topk=2,
        )

    def test_forward_enforces_topk_activations(self) -> None:
        model = TopKSparseAutoencoderModel(self.config)
        outputs = model(inputs=self.inputs)
        nonzero_counts = (outputs.latents != 0).sum(dim=-1)
        self.assertTrue(torch.equal(nonzero_counts, torch.full_like(nonzero_counts, 2)))
        self.assertIn("topk_sparsity", outputs.loss_dict)

    def test_save_and_load_pretrained_round_trip(self) -> None:
        model = TopKSparseAutoencoderModel(self.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = TopKSparseAutoencoderModel.from_pretrained(tmpdir)
        self.assertEqual(loaded.config.topk, 2)


if __name__ == "__main__":
    unittest.main()
