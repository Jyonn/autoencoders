# Variational Family

## Base tree

```text
BaseVariationalAutoencoderConfig
├── BaseAutoencoderConfig
│   ├── latent_dim
│   └── reconstruction_loss
├── kl_weight
├── free_bits
├── kl_warmup_epochs
├── kl_start_weight
└── use_mean_in_eval
```

## Models

```text
VariationalAutoencoderConfig
└── no additional model fields

BetaVariationalAutoencoderConfig
└── inherits base variational fields; beta behavior is expressed through kl_weight

DenoisingVariationalAutoencoderConfig
├── noise_type
├── noise_std
└── masking_ratio

HierarchicalVariationalAutoencoderConfig
└── top_latent_dim

VampPriorVariationalAutoencoderConfig
└── num_pseudo_inputs

InformationVariationalAutoencoderConfig
├── mmd_weight
└── kernel_bandwidths

MMDVariationalAutoencoderConfig
├── mmd_weight
└── kernel_bandwidths

DIPVariationalAutoencoderConfig
├── dip_type
├── lambda_diag
└── lambda_offdiag

BetaTCVariationalAutoencoderConfig
└── tc_weight

FactorVariationalAutoencoderConfig
├── tc_weight
└── discriminator_hidden_dims
```

## Notes

- Variational models often change decoder runtime input space from encoder output space.
- Because of that, `decoder: null` is intentionally rejected for many VAE-family runs unless the specs still match exactly.
