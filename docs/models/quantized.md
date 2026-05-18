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
- Hierarchical models such as `VQVAE2` have decoder spaces that differ from encoder output spaces, so they should use explicit decoders.
- `FSQ` and `RFSQ` use fixed scalar levels rather than learned vector codebooks, so their dead-code handling semantics differ from `VQVAE`, `PQVAE`, and `RQVAE`.
