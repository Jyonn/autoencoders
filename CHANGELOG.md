# Changelog

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
