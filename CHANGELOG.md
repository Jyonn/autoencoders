# Changelog

## 0.4.1 - 2026-05-19

- Added a dedicated MkDocs configuration reference page that explains the meaning of dataset, model, backbone, and trainer parameters used by the unified YAML entrypoint.
- Linked the dataset, backbone, training, and model overview pages back to the parameter reference so configuration fields are easier to discover.
- Fixed the `RQVAE` EMA residual-codebook update path so later residual quantizers continue to use the same forward-pass residual chain that produced the recorded indices.

## 0.4.0 - 2026-05-18

- Added trainer-side optimization features inspired by `RQ-VAE`, including `AdamW`, weight decay, schedulers, warmup, and gradient clipping through YAML-configurable training settings.
- Extended `MLPModule` with dropout, optional normalization (`layernorm` and `batchnorm`), and configurable Xavier initialization for stronger embedding autoencoder experiments.
- Added k-means codebook initialization for learned vector-codebook quantizers across `VQVAE`, `PQVAE`, `RQVAE`, `GumbelVQ`, and `VQVAE2`.
- Added Sinkhorn assignment as a learned-codebook quantization strategy, including per-codebook epsilon lists with `0.0` slots falling back to nearest-neighbor assignment.
- Expanded model documentation and MkDocs coverage for backbone, training, and quantized-model configuration trees.
- Refreshed stronger example YAMLs for `glove` experiments to exercise the new MLP regularization and quantizer assignment features.

## 0.3.0 - 2026-05-18

- Added `VisionTransformerModule` for image-to-sequence latent backbones on `H x W x C` image specs.
- Tightened auto-inferred decoder semantics: `decoder=None` is now allowed only when the model decoder input spec matches the encoder output spec.
- Aligned AE-, VAE-, and quantized-model core-space flows around explicit `sample_spec`, `encoder_output_spec`, and `core_spec` transitions.
- Added and refreshed release-ready YAML configs for `glove` and `cifar10` experiments, including CNN, ViT, and quantized image runs.
- Improved trainer-side config printing and pipeline shape tracing for YAML-driven experiments.
- Hardened CIFAR-10 downloading and quantized trainer compatibility for `RFSQ`, `VampPriorVAE`, `GumbelVQ`, and `VQVAE2`.

## 0.2.0 - 2026-05-18

- Unified training under one YAML-first entrypoint at `examples/trainer.py`.
- Added `DataSpec`-driven backbone construction and shape tracing.
- Added built-in `CNNModule` and `CIFAR10Dataset`.
- Aligned AE, VAE, and quantized families around shared core-space projection semantics.
- Added release-ready example configs for `glove` and `cifar10` experiments.
