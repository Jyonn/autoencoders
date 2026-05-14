"""Output containers used by autoencoder-family models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BaseAutoencoderOutput:
    """Standard return object for deterministic autoencoder models."""

    loss: Any = None
    reconstruction: Any = None
    latents: Any = None
    encoded: Any = None
    reconstruction_loss: Any = None
    hidden_states: dict[str, Any] | None = None
    loss_dict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DenoisingAutoencoderOutput(BaseAutoencoderOutput):
    """Output for denoising autoencoders."""


@dataclass
class SparseAutoencoderOutput(BaseAutoencoderOutput):
    """Output for sparse autoencoders."""

    sparsity_loss: Any = None


@dataclass
class TopKSparseAutoencoderOutput(BaseAutoencoderOutput):
    """Output for Top-K sparse autoencoders."""

    topk_sparsity: Any = None


@dataclass
class KLSparseAutoencoderOutput(BaseAutoencoderOutput):
    """Output for KL-sparse autoencoders."""

    kl_sparsity_loss: Any = None


@dataclass
class ContractiveAutoencoderOutput(BaseAutoencoderOutput):
    """Output for contractive autoencoders."""

    contractive_loss: Any = None


@dataclass
class VariationalAutoencoderOutput(BaseAutoencoderOutput):
    """Standard return object for variational autoencoder models."""

    posterior_mean: Any = None
    posterior_logvar: Any = None
    kl_loss: Any = None
    free_bits_kl_loss: Any = None
    effective_kl_weight: Any = None


@dataclass
class DenoisingVariationalAutoencoderOutput(VariationalAutoencoderOutput):
    """Output for denoising variational autoencoders."""


@dataclass
class HierarchicalVariationalAutoencoderOutput(VariationalAutoencoderOutput):
    """Output for hierarchical variational autoencoders."""

    hierarchical_kl_loss: Any = None


@dataclass
class WassersteinAutoencoderOutput(BaseAutoencoderOutput):
    """Output for Wasserstein autoencoders."""

    mmd_loss: Any = None


@dataclass
class AdversarialAutoencoderOutput(BaseAutoencoderOutput):
    """Output for adversarial autoencoders."""

    adversarial_loss: Any = None
    discriminator_loss: Any = None


@dataclass
class QuantizedAutoencoderOutput(BaseAutoencoderOutput):
    """Standard return object for quantized autoencoder models."""

    quantized_latents: Any = None
    codebook_indices: Any = None
    commitment_loss: Any = None
    codebook_loss: Any = None


@dataclass
class FiniteScalarQuantizedAutoencoderOutput(QuantizedAutoencoderOutput):
    """Output for finite scalar quantized autoencoders."""


# Backward-compatible alias for older integrations that imported AutoencoderOutput.
AutoencoderOutput = BaseAutoencoderOutput


@dataclass
class AutoencoderExport:
    """Standard export object for model-produced latent artifacts."""

    model_type: str
    latents: Any = None
    reconstruction: Any = None
    encoded: Any = None
    posterior_mean: Any = None
    posterior_logvar: Any = None
    quantized_latents: Any = None
    codebook_indices: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
