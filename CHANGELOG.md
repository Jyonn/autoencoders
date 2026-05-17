# Changelog

## 0.2.0 - 2026-05-18

- Unified training under one YAML-first entrypoint at `examples/trainer.py`.
- Added `DataSpec`-driven backbone construction and shape tracing.
- Added built-in `CNNModule` and `CIFAR10Dataset`.
- Aligned AE, VAE, and quantized families around shared core-space projection semantics.
- Added release-ready example configs for `glove` and `cifar10` experiments.
