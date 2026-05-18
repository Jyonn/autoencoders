# Quantized Family

## Base tree

```text
BaseVectorQuantizedAutoencoderConfig
├── BaseAutoencoderConfig
│   ├── latent_dim
│   └── reconstruction_loss
├── codebook_size
├── commitment_weight
├── codebook_weight
├── assignment_strategy
├── sinkhorn_epsilon
├── sinkhorn_iters
├── kmeans_init
├── kmeans_iters
├── use_ema_codebook
├── ema_decay
├── ema_epsilon
└── dead_code_reset
```

## Models

```text
VectorQuantizedAutoencoderConfig
└── no additional model fields

GumbelQuantizedAutoencoderConfig
├── temperature
├── min_temperature
└── anneal_rate

FiniteScalarQuantizedAutoencoderConfig
└── num_levels

ResidualFiniteScalarQuantizedAutoencoderConfig
├── num_levels
└── num_quantizers

ProductQuantizedAutoencoderConfig
└── num_codebooks

ResidualQuantizedAutoencoderConfig
└── num_quantizers

HierarchicalVectorQuantizedAutoencoderConfig
└── top_latent_dim
```

## Notes

- Quantized models depend heavily on codebook initialization, usage balance, and reconstruction-vs-quantization weighting.
- `kmeans_init: true` initializes learned codebooks from the first training batch of encoder latents instead of uniform random weights.
- `assignment_strategy: sinkhorn` switches learned vector-codebook quantizers from nearest-neighbor assignment to balanced Sinkhorn assignment. `sinkhorn_epsilon` may be a single value or a per-codebook list. Any slot set to `0.0` falls back to nearest-neighbor assignment for that codebook, which matches the original `RQ-VAE` semantics.
- Hierarchical models such as `VQVAE2` have decoder spaces that differ from encoder output spaces, so they should use explicit decoders.
- `FSQ` and `RFSQ` use fixed scalar levels rather than learned vector codebooks, so their dead-code handling semantics differ from `VQVAE`, `PQVAE`, and `RQVAE`.
