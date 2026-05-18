# Deterministic AE Family

## Base tree

```text
BaseAutoencoderConfig
├── latent_dim
└── reconstruction_loss
```

## Models

```text
AutoencoderConfig
└── no additional model fields

DenoisingAutoencoderConfig
├── noise_type
├── noise_std
└── masking_ratio

ContractiveAutoencoderConfig
└── contractive_weight

SparseAutoencoderConfig
├── sparsity_weight
└── target_activation

TopKSparseAutoencoderConfig
└── topk

KLSparseAutoencoderConfig
├── sparsity_weight
└── target_activation

WassersteinAutoencoderConfig
├── mmd_weight
└── kernel_bandwidths

AdversarialAutoencoderConfig
├── adversarial_weight
└── discriminator_hidden_dims
```

## Notes

- Deterministic families usually allow `decoder: null` when reversing the encoder preserves the decoder runtime input spec.
- Architecture-specific fields belong in encoder and decoder module configs rather than here.
