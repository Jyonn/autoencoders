"""Output containers used by autoencoder-family models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AutoencoderOutput:
    """Standard return object for basic autoencoder models."""

    loss: Any = None
    reconstruction: Any = None
    latents: Any = None
    encoded: Any = None
    posterior_mean: Any = None
    posterior_logvar: Any = None
    reconstruction_loss: Any = None
    kl_loss: Any = None
    hidden_states: dict[str, Any] | None = None
    loss_dict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
