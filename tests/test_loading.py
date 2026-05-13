"""Tests for dataset and model loading helpers."""

from __future__ import annotations

import unittest

from autoencoders import load_dataset, load_model
from autoencoders.data import GloVeDataset
from autoencoders.models.ae.modeling_ae import AutoencoderModel
from autoencoders.models.dae.modeling_dae import DenoisingAutoencoderModel


class LoadingHelpersTest(unittest.TestCase):
    def test_load_dataset_returns_glove(self) -> None:
        dataset = load_dataset("glove", dim=50, max_vectors=32)
        self.assertIsInstance(dataset, GloVeDataset)

    def test_load_model_returns_autoencoder(self) -> None:
        model = load_model("ae", input_dim=16, latent_dim=4, hidden_dims=[8])
        self.assertIsInstance(model, AutoencoderModel)

    def test_load_model_returns_denoising_autoencoder(self) -> None:
        model = load_model(
            "dae",
            input_dim=16,
            latent_dim=4,
            hidden_dims=[8],
            noise_type="gaussian",
            noise_std=0.2,
        )
        self.assertIsInstance(model, DenoisingAutoencoderModel)


if __name__ == "__main__":
    unittest.main()
