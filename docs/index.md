# autoencoders

`autoencoders` is a PyTorch-first library for deterministic, variational, and quantized autoencoder-family models with one consistent config, module, trainer, and checkpoint surface.

## What this documentation covers

- how to build models from `sample_spec`
- how dataset, encoder, core, and decoder specs fit together
- how to train from one YAML-first entrypoint
- how different model families extend the shared base configs

## Core ideas

- Datasets expose canonical `DataSpec` objects.
- Backbones are explicit and are built from those specs.
- Models distinguish:
  - `sample_spec`
  - `encoder_output_spec`
  - `core_spec`
- `decoder: null` is intentionally strict and only works when reversing the encoder produces a decoder whose runtime input spec matches the model's declared decoder input spec.

## Quick links

- [Getting Started](getting-started.md)
- [Backbones](backbones.md)
- [Training](training.md)
- [Model parameter trees](models/index.md)
