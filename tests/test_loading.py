"""Tests for dataset and model loading helpers."""

from __future__ import annotations

import unittest

from autoencoders import load_dataset, load_model
from autoencoders.data import GloVeDataset
from autoencoders.models.ae.modeling_ae import AutoencoderModel
from autoencoders.models.betavae.modeling_betavae import BetaVariationalAutoencoderModel
from autoencoders.models.dae.modeling_dae import DenoisingAutoencoderModel
from autoencoders.models.sae.modeling_sae import SparseAutoencoderModel
from autoencoders.models.vae.modeling_vae import VariationalAutoencoderModel


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

    def test_load_model_returns_sparse_autoencoder(self) -> None:
        model = load_model("sae", input_dim=16, latent_dim=4, hidden_dims=[8], sparsity_weight=0.01)
        self.assertIsInstance(model, SparseAutoencoderModel)

    def test_load_model_returns_variational_autoencoder(self) -> None:
        model = load_model("vae", input_dim=16, latent_dim=4, hidden_dims=[8], kl_weight=0.5)
        self.assertIsInstance(model, VariationalAutoencoderModel)

    def test_load_model_returns_beta_variational_autoencoder(self) -> None:
        model = load_model("betavae", input_dim=16, latent_dim=4, hidden_dims=[8], beta=4.0)
        self.assertIsInstance(model, BetaVariationalAutoencoderModel)


if __name__ == "__main__":
    unittest.main()
