"""Tests for dataset and model loading helpers."""

from __future__ import annotations

import unittest

from autoencoders import get_module_modules, load_dataset, load_model
from autoencoders.models.aae.modeling_aae import AdversarialAutoencoderModel
from autoencoders.data import (
    ConceptNetNumberbatchDataset,
    FastTextEnglishDataset,
    Flickr30kDataset,
    GloVeDataset,
    MultiNLIDataset,
    SNLIDataset,
)
from autoencoders.models.ae.modeling_ae import AutoencoderModel
from autoencoders.models.betavae.modeling_betavae import BetaVariationalAutoencoderModel
from autoencoders.models.betatcvae.modeling_betatcvae import BetaTCVariationalAutoencoderModel
from autoencoders.models.cae.modeling_cae import ContractiveAutoencoderModel
from autoencoders.models.dae.modeling_dae import DenoisingAutoencoderModel
from autoencoders.models.dipvae.modeling_dipvae import DIPVariationalAutoencoderModel
from autoencoders.models.dvae.modeling_dvae import DenoisingVariationalAutoencoderModel
from autoencoders.models.factorvae.modeling_factorvae import FactorVariationalAutoencoderModel
from autoencoders.models.fsq.modeling_fsq import FiniteScalarQuantizedAutoencoderModel
from autoencoders.models.gumbelvq.modeling_gumbelvq import GumbelQuantizedAutoencoderModel
from autoencoders.models.hvae.modeling_hvae import HierarchicalVariationalAutoencoderModel
from autoencoders.models.infovae.modeling_infovae import InformationVariationalAutoencoderModel
from autoencoders.models.klsae.modeling_klsae import KLSparseAutoencoderModel
from autoencoders.models.mmdvae.modeling_mmdvae import MMDVariationalAutoencoderModel
from autoencoders.models.pqvae.modeling_pqvae import ProductQuantizedAutoencoderModel
from autoencoders.models.rfsq.modeling_rfsq import ResidualFiniteScalarQuantizedAutoencoderModel
from autoencoders.models.rqvae.modeling_rqvae import ResidualQuantizedAutoencoderModel
from autoencoders.models.sae.modeling_sae import SparseAutoencoderModel
from autoencoders.models.topksae.modeling_topksae import TopKSparseAutoencoderModel
from autoencoders.models.vae.modeling_vae import VariationalAutoencoderModel
from autoencoders.models.vamppriorvae.modeling_vamppriorvae import VampPriorVariationalAutoencoderModel
from autoencoders.models.wae.modeling_wae import WassersteinAutoencoderModel
from autoencoders.models.vqvae2.modeling_vqvae2 import HierarchicalVectorQuantizedAutoencoderModel
from autoencoders.models.loading import get_model_modules
from autoencoders.models.vqvae.modeling_vqvae import VectorQuantizedAutoencoderModel


class LoadingHelpersTest(unittest.TestCase):
    def test_load_dataset_returns_glove(self) -> None:
        dataset = load_dataset("glove", dim=50, max_vectors=32)
        self.assertIsInstance(dataset, GloVeDataset)

    def test_load_dataset_returns_fasttext(self) -> None:
        dataset = load_dataset("fasttext", dim=300, max_vectors=32)
        self.assertIsInstance(dataset, FastTextEnglishDataset)

    def test_load_dataset_returns_numberbatch(self) -> None:
        dataset = load_dataset("numberbatch", dim=300, max_vectors=32)
        self.assertIsInstance(dataset, ConceptNetNumberbatchDataset)

    def test_load_dataset_returns_snli(self) -> None:
        dataset = load_dataset("snli", max_vectors=32)
        self.assertIsInstance(dataset, SNLIDataset)

    def test_load_dataset_returns_multinli(self) -> None:
        dataset = load_dataset("multinli", max_vectors=32)
        self.assertIsInstance(dataset, MultiNLIDataset)

    def test_load_dataset_returns_flickr30k(self) -> None:
        dataset = load_dataset("flickr30k", max_vectors=32)
        self.assertIsInstance(dataset, Flickr30kDataset)

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

    def test_load_model_returns_topk_sparse_autoencoder(self) -> None:
        model = load_model("topksae", input_dim=16, latent_dim=4, hidden_dims=[8], topk=2)
        self.assertIsInstance(model, TopKSparseAutoencoderModel)

    def test_load_model_returns_kl_sparse_autoencoder(self) -> None:
        model = load_model("klsae", input_dim=16, latent_dim=4, hidden_dims=[8], sparsity_weight=0.01, target_activation=0.05)
        self.assertIsInstance(model, KLSparseAutoencoderModel)

    def test_load_model_returns_variational_autoencoder(self) -> None:
        model = load_model("vae", input_dim=16, latent_dim=4, hidden_dims=[8], kl_weight=0.5)
        self.assertIsInstance(model, VariationalAutoencoderModel)

    def test_load_model_returns_denoising_variational_autoencoder(self) -> None:
        model = load_model("dvae", input_dim=16, latent_dim=4, hidden_dims=[8], kl_weight=0.5)
        self.assertIsInstance(model, DenoisingVariationalAutoencoderModel)

    def test_load_model_returns_beta_variational_autoencoder(self) -> None:
        model = load_model("betavae", input_dim=16, latent_dim=4, hidden_dims=[8], beta=4.0)
        self.assertIsInstance(model, BetaVariationalAutoencoderModel)

    def test_load_model_returns_beta_tc_variational_autoencoder(self) -> None:
        model = load_model("betatcvae", input_dim=16, latent_dim=4, hidden_dims=[8], total_correlation_weight=6.0)
        self.assertIsInstance(model, BetaTCVariationalAutoencoderModel)

    def test_load_model_returns_hierarchical_variational_autoencoder(self) -> None:
        model = load_model("hvae", input_dim=16, latent_dim=4, hidden_dims=[8], top_latent_dim=2)
        self.assertIsInstance(model, HierarchicalVariationalAutoencoderModel)

    def test_load_model_returns_information_variational_autoencoder(self) -> None:
        model = load_model("infovae", input_dim=16, latent_dim=4, hidden_dims=[8], mmd_weight=5.0)
        self.assertIsInstance(model, InformationVariationalAutoencoderModel)

    def test_load_model_returns_mmd_variational_autoencoder(self) -> None:
        model = load_model("mmdvae", input_dim=16, latent_dim=4, hidden_dims=[8], mmd_weight=10.0)
        self.assertIsInstance(model, MMDVariationalAutoencoderModel)

    def test_load_model_returns_vampprior_variational_autoencoder(self) -> None:
        model = load_model("vamppriorvae", input_dim=16, latent_dim=4, hidden_dims=[8], num_pseudo_inputs=32)
        self.assertIsInstance(model, VampPriorVariationalAutoencoderModel)

    def test_load_model_returns_factor_variational_autoencoder(self) -> None:
        model = load_model("factorvae", input_dim=16, latent_dim=4, hidden_dims=[8], tc_weight=10.0)
        self.assertIsInstance(model, FactorVariationalAutoencoderModel)

    def test_load_model_returns_dip_variational_autoencoder(self) -> None:
        model = load_model("dipvae", input_dim=16, latent_dim=4, hidden_dims=[8], dip_weight=10.0)
        self.assertIsInstance(model, DIPVariationalAutoencoderModel)

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

    def test_load_model_returns_gumbel_quantized_autoencoder(self) -> None:
        model = load_model("gumbelvq", input_dim=16, latent_dim=4, hidden_dims=[8], codebook_size=32)
        self.assertIsInstance(model, GumbelQuantizedAutoencoderModel)

    def test_load_model_returns_finite_scalar_quantized_autoencoder(self) -> None:
        model = load_model("fsq", input_dim=16, latent_dim=4, hidden_dims=[8], num_levels=8)
        self.assertIsInstance(model, FiniteScalarQuantizedAutoencoderModel)

    def test_load_model_returns_residual_finite_scalar_quantized_autoencoder(self) -> None:
        model = load_model("rfsq", input_dim=16, latent_dim=4, hidden_dims=[8], num_levels=8, num_quantizers=2)
        self.assertIsInstance(model, ResidualFiniteScalarQuantizedAutoencoderModel)

    def test_load_model_returns_product_quantized_autoencoder(self) -> None:
        model = load_model("pqvae", input_dim=16, latent_dim=4, hidden_dims=[8], codebook_size=16, num_codebooks=2)
        self.assertIsInstance(model, ProductQuantizedAutoencoderModel)

    def test_load_model_returns_residual_quantized_autoencoder(self) -> None:
        model = load_model("rqvae", input_dim=16, latent_dim=4, hidden_dims=[8], codebook_size=16, num_quantizers=2)
        self.assertIsInstance(model, ResidualQuantizedAutoencoderModel)

    def test_load_model_returns_hierarchical_vector_quantized_autoencoder(self) -> None:
        model = load_model("vqvae2", input_dim=16, latent_dim=4, hidden_dims=[8], codebook_size=16, top_latent_dim=3)
        self.assertIsInstance(model, HierarchicalVectorQuantizedAutoencoderModel)

    def test_model_module_discovery_excludes_internal_base_namespace(self) -> None:
        model_modules = get_model_modules()
        self.assertNotIn("base", model_modules)
        self.assertIn("ae", model_modules)
        self.assertIn("vqvae", model_modules)

    def test_backbone_module_discovery_excludes_internal_base_namespace(self) -> None:
        module_modules = get_module_modules()
        self.assertNotIn("base", module_modules)
        self.assertIn("mlp", module_modules)


if __name__ == "__main__":
    unittest.main()
