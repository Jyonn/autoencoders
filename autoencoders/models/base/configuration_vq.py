"""Base configuration shared by vector-quantized autoencoder families."""

from __future__ import annotations

from collections.abc import Sequence

from ..ae.configuration_ae import AutoencoderConfig


class BaseVectorQuantizedAutoencoderConfig(AutoencoderConfig):
    """Base config for VQ-style autoencoders."""

    model_type = "base_vector_quantized_autoencoder"

    def __init__(
        self,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        assignment_strategy: str = "nearest",
        sinkhorn_epsilon: float = 0.0,
        sinkhorn_iters: int = 100,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        use_ema_codebook: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_reset: bool = False,
        dead_code_threshold: int = 0,
        **kwargs,
    ) -> None:
        if codebook_size <= 0:
            raise ValueError("codebook_size must be a positive integer.")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative.")
        if codebook_weight < 0:
            raise ValueError("codebook_weight must be non-negative.")
        if assignment_strategy not in {"nearest", "sinkhorn"}:
            raise ValueError("assignment_strategy must be either 'nearest' or 'sinkhorn'.")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be a positive integer.")
        if kmeans_iters <= 0:
            raise ValueError("kmeans_iters must be a positive integer.")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in the range [0, 1).")
        if ema_epsilon <= 0:
            raise ValueError("ema_epsilon must be positive.")
        if dead_code_threshold < 0:
            raise ValueError("dead_code_threshold must be non-negative.")
        if assignment_strategy == "sinkhorn" and dead_code_reset:
            raise ValueError("dead_code_reset cannot be enabled together with assignment_strategy='sinkhorn'.")
        if kwargs.get("latent_dim") is None:
            raise TypeError("BaseVectorQuantizedAutoencoderConfig requires `latent_dim`.")
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.assignment_strategy = assignment_strategy
        self.sinkhorn_epsilon = self._validate_sinkhorn_epsilon(sinkhorn_epsilon)
        self.sinkhorn_iters = sinkhorn_iters
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.use_ema_codebook = use_ema_codebook
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.dead_code_reset = dead_code_reset
        self.dead_code_threshold = dead_code_threshold
        super().__init__(**kwargs)

    @staticmethod
    def _validate_sinkhorn_epsilon(sinkhorn_epsilon: float | Sequence[float]) -> float | list[float]:
        if isinstance(sinkhorn_epsilon, Sequence) and not isinstance(sinkhorn_epsilon, (str, bytes)):
            epsilon_values = [float(epsilon) for epsilon in sinkhorn_epsilon]
            if any(epsilon < 0 for epsilon in epsilon_values):
                raise ValueError("sinkhorn_epsilon values must be non-negative.")
            return epsilon_values
        epsilon_value = float(sinkhorn_epsilon)
        if epsilon_value < 0:
            raise ValueError("sinkhorn_epsilon must be non-negative.")
        return epsilon_value

    def validate_sinkhorn_slot_count(self, expected_count: int, label: str) -> None:
        if not isinstance(self.sinkhorn_epsilon, list):
            return
        if len(self.sinkhorn_epsilon) not in {1, expected_count}:
            raise ValueError(
                f"{label} expects sinkhorn_epsilon to provide either 1 value or {expected_count} values, "
                f"got {len(self.sinkhorn_epsilon)}."
            )
