"""Tests for dataset and model loading helpers."""

from __future__ import annotations

import unittest

from autoencoders import load_dataset, load_model
from autoencoders.models.aae.modeling_aae import AdversarialAutoencoderModel
from autoencoders.data import GloVeDataset
from autoencoders.models.ae.modeling_ae import AutoencoderModel
from autoencoders.models.betavae.modeling_betavae import BetaVariationalAutoencoderModel
from autoencoders.models.cae.modeling_cae import ContractiveAutoencoderModel
from autoencoders.models.dae.modeling_dae import DenoisingAutoencoderModel
from autoencoders.models.pqvae.modeling_pqvae import ProductQuantizedAutoencoderModel
from autoencoders.models.rqvae.modeling_rqvae import ResidualQuantizedAutoencoderModel
from autoencoders.models.sae.modeling_sae import SparseAutoencoderModel
from autoencoders.models.vae.modeling_vae import VariationalAutoencoderModel
from autoencoders.models.wae.modeling_wae import WassersteinAutoencoderModel
from autoencoders.models.vqvae.modeling_vqvae import VectorQuantizedAutoencoderModel


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

    def test_load_model_returns_contractive_autoencoder(self) -> None:
        model = load_model("cae", input_dim=16, latent_dim=4, hidden_dims=[8], contractive_weight=0.05)
        self.assertIsInstance(model, ContractiveAutoencoderModel)

    def test_load_model_returns_sparse_autoencoder(self) -> None:
        model = load_model("sae", input_dim=16, latent_dim=4, hidden_dims=[8], sparsity_weight=0.01)
        self.assertIsInstance(model, SparseAutoencoderModel)

    def test_load_model_returns_variational_autoencoder(self) -> None:
        model = load_model("vae", input_dim=16, latent_dim=4, hidden_dims=[8], kl_weight=0.5)
        self.assertIsInstance(model, VariationalAutoencoderModel)

    def test_load_model_returns_beta_variational_autoencoder(self) -> None:
        model = load_model("betavae", input_dim=16, latent_dim=4, hidden_dims=[8], beta=4.0)
        self.assertIsInstance(model, BetaVariationalAutoencoderModel)

    def test_load_model_returns_wasserstein_autoencoder(self) -> None:
        model = load_model("wae", input_dim=16, latent_dim=4, hidden_dims=[8], mmd_weight=5.0)
        self.assertIsInstance(model, WassersteinAutoencoderModel)

    def test_load_model_returns_adversarial_autoencoder(self) -> None:
        model = load_model(
            "aae",
            input_dim=16,
            latent_dim=4,
            hidden_dims=[8],
            discriminator_hidden_dims=[6],
            adversarial_weight=0.5,
        )
        self.assertIsInstance(model, AdversarialAutoencoderModel)

    def test_load_model_returns_vector_quantized_autoencoder(self) -> None:
        model = load_model("vqvae", input_dim=16, latent_dim=4, hidden_dims=[8], codebook_size=32)
        self.assertIsInstance(model, VectorQuantizedAutoencoderModel)

    def test_load_model_returns_product_quantized_autoencoder(self) -> None:
        model = load_model("pqvae", input_dim=16, latent_dim=4, hidden_dims=[8], codebook_size=16, num_codebooks=2)
        self.assertIsInstance(model, ProductQuantizedAutoencoderModel)

    def test_load_model_returns_residual_quantized_autoencoder(self) -> None:
        model = load_model("rqvae", input_dim=16, latent_dim=4, hidden_dims=[8], codebook_size=16, num_quantizers=2)
        self.assertIsInstance(model, ResidualQuantizedAutoencoderModel)


if __name__ == "__main__":
    unittest.main()
