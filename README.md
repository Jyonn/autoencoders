<div align="center">

# autoencoders

**A latent-model toolkit for deterministic, variational, and quantized autoencoders**

<p>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/model_families-20%2B-111827" alt="20+ model families" />
  <img src="https://img.shields.io/badge/datasets-glove%20%7C%20fasttext%20%7C%20numberbatch%20%7C%20snli%20%7C%20multinli%20%7C%20flickr30k%20%7C%20cifar10-0F766E" alt="Datasets" />
  <img src="https://img.shields.io/badge/checkpoints-save__pretrained%20%2F%20from__pretrained-7C3AED" alt="Checkpoint API" />
</p>

<p>
  <strong>Build, train, serialize, and export latent models with one consistent API.</strong>
</p>

</div>

`autoencoders` is a PyTorch-first library for autoencoder-family models across deterministic, variational, and quantized latent spaces.

The project goal is simple: make autoencoders feel composable, serializable, and reusable in the same way `transformers` did for sequence models.

## Why autoencoders

<table>
  <tr>
    <td valign="top" width="25%">
      <strong>🧩 Unified API</strong><br />
      One package shape across `AE`, `VAE`, `VQ-VAE`, `PQ-VAE`, `RQ-VAE`, `WAE`, `AAE`, and more.
    </td>
    <td valign="top" width="25%">
      <strong>🧠 Latent-first design</strong><br />
      Treat reconstruction, posterior statistics, quantized codes, and exported latents as first-class outputs.
    </td>
    <td valign="top" width="25%">
      <strong>📦 Reusable checkpoints</strong><br />
      Use `save_pretrained()` and `from_pretrained()` for stable, shareable model artifacts.
    </td>
    <td valign="top" width="25%">
      <strong>🚀 Real training flow</strong><br />
      Ship with trainers, datasets, shell wrappers, and packaging hooks for end-to-end experiments.
    </td>
  </tr>
</table>

## What It Covers

Current model families include:

- Deterministic models: `AE`, `DAE`, `CAE`, `SAE`, `TopKSAE`, `KLSAE`, `WAE`, `AAE`
- Variational models: `VAE`, `DVAE`, `BetaVAE`, `BetaTCVAE`, `DIPVAE`, `InfoVAE`, `MMDVAE`, `FactorVAE`, `VampPriorVAE`, `HVAE`
- Quantized models: `VQVAE`, `GumbelVQ`, `FSQ`, `RFSQ`, `PQVAE`, `RQVAE`, `VQVAE2`

Core interfaces include:

- `Config + Model + Output + Export`
- `save_pretrained()` / `from_pretrained()`
- `encode()` / `decode()` / `reconstruct()` / `export()`
- family-specific trainers for deterministic, variational, and quantized models

## At a Glance

<table>
  <tr>
    <th align="left">Family</th>
    <th align="left">Examples</th>
    <th align="left">Key outputs</th>
  </tr>
  <tr>
    <td><strong>Deterministic</strong></td>
    <td>`AE`, `DAE`, `CAE`, `SAE`, `TopKSAE`, `KLSAE`</td>
    <td>`reconstruction`, `latents`, sparse and contractive penalties</td>
  </tr>
  <tr>
    <td><strong>Variational</strong></td>
    <td>`VAE`, `DVAE`, `BetaVAE`, `HVAE`</td>
    <td>`posterior_mean`, `posterior_logvar`, `kl_loss`, `free_bits_kl_loss`</td>
  </tr>
  <tr>
    <td><strong>Quantized</strong></td>
    <td>`VQVAE`, `FSQ`, `PQVAE`, `RQVAE`</td>
    <td>`quantized_latents`, `codebook_indices`, usage and perplexity metrics</td>
  </tr>
</table>

## Installation

Install the package:

```bash
pip install autoencoders
```

Install with PyTorch dependencies:

```bash
pip install "autoencoders[torch]"
```

Install with encoder-backed text dataset support:

```bash
pip install "autoencoders[text]"
```

Install with CLIP-backed multimodal dataset support:

```bash
pip install "autoencoders[clip]"
```

Install everything commonly needed for experiments:

```bash
pip install "autoencoders[all]"
```

If you are working from source and plan to build or publish packages:

```bash
pip install "autoencoders[dev]"
```

## Quick Start

Build a basic `AE + MLP` model explicitly from a sample spec:

```python
import torch

from autoencoders import AutoencoderConfig, AutoencoderModel
from autoencoders.data.base import TensorSpec

model = AutoencoderModel(
    config=AutoencoderConfig(latent_dim=16),
    sample_spec=TensorSpec(shape=(50,)),
    encoder="mlp",
    encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
    decoder="mlp",
    decoder_config={"hidden_dims": [64, 50], "activation": "relu", "use_bias": True},
)

inputs = torch.randn(32, 50)
outputs = model(inputs)

print(outputs.loss)
print(outputs.latents.shape)
print(outputs.reconstruction.shape)
```

Save and load checkpoints:

```python
model.save_pretrained("artifacts/ae")
restored = AutoencoderModel.from_pretrained("artifacts/ae")
```

Inspect the model pipeline and layer-by-layer shape trace:

```python
for step in model.get_pipeline_trace():
    print(step.name, "->", step.output_spec)
```

Train from YAML:

```bash
python examples/trainer.py --config examples/configs/glove/ae.yaml --epoch 5
```

## Product Surface

Use the package at three different layers:

- `Model layer`: build or load latent models with typed configs
- `Training layer`: train deterministic, variational, or quantized families with dedicated trainers
- `Experiment layer`: run reusable YAML configs with one trainer entrypoint on real datasets

## Model Loading

Load a model dynamically by name while still keeping backbone selection explicit:

```python
from autoencoders import load_model
from autoencoders.data.base import TensorSpec

model = load_model(
    "vae",
    sample_spec=TensorSpec(shape=(50,)),
    latent_dim=16,
    kl_weight=0.1,
    free_bits=0.02,
    encoder="mlp",
    encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
    decoder="mlp",
    decoder_config={"hidden_dims": [64, 50], "activation": "relu", "use_bias": True},
)
```

## Datasets

The library currently ships with embedding-first datasets plus one image dataset for CNN- and ViT-backed experiments:

- `glove`
- `fasttext`
- `numberbatch`
- `snli`
- `multinli`
- `flickr30k`
- `cifar10`

Load a dataset directly:

```python
from autoencoders import load_dataset

dataset = load_dataset("glove", dim=50, max_vectors=50000)
loaders = dataset.get_dataloaders(batch_size=256)
```

Encoder-backed sentence datasets materialize embeddings during `prepare()` and cache the result just like static embedding tables:

```python
dataset = load_dataset(
    "snli",
    encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    max_vectors=50000,
)
loaders = dataset.get_dataloaders(batch_size=256)
```

CLIP-backed multimodal datasets follow the same cached artifact pattern:

```python
dataset = load_dataset(
    "flickr30k",
    encoder_name="ViT-B-32",
    encoder_pretrained="laion2b_s34b_b79k",
    modality="both",
    max_vectors=50000,
)
loaders = dataset.get_dataloaders(batch_size=256)
```

Image data uses `H x W x C` specs end to end:

```python
dataset = load_dataset("cifar10", max_examples=10000)
print(dataset.get_sample_spec())  # TensorSpec(shape=(32, 32, 3))
```

## Backbone Semantics

Backbones are configured explicitly and built from the dataset-driven `sample_spec`.

- `MLPModule` consumes tensor specs whose last dimension is the feature width.
- `CNNModule` consumes image-like `TensorSpec(shape=(H, W, C))` values and handles `HWC <-> NCHW` conversion internally.
- `VisionTransformerModule` also consumes image-like `TensorSpec(shape=(H, W, C))`, patchifies them internally, and exposes sequence-shaped latent specs.

Auto-inferred decoders are intentionally strict:

- `decoder: null` is supported only when reversing the encoder produces a decoder whose runtime input spec matches the model's decoder input spec.
- Models whose decoder space differs from encoder output space, such as hierarchical or latent-shape-changing variants, must provide an explicit decoder config.

For explicit image decoders, use `transpose: true` when you want an upsampling transposed-convolution stack:

```yaml
decoder:
  name: cnn
  config:
    channels: [64, 3]
    kernel_sizes: [4, 4]
    strides: [2, 2]
    paddings: [1, 1]
    activation: relu
    use_bias: true
    transpose: true
```

Downloaded datasets use a global cache:

- default: `~/.cache/autoencoders`
- override with: `AUTOENCODERS_CACHE=/your/cache/path`

This makes the package useful both as:

- a standalone training library
- a latent-model subsystem inside larger PyTorch projects

## Training API

Deterministic training:

```python
from autoencoders import AETrainer, TrainingConfig

trainer = AETrainer(
    model=model,
    args=TrainingConfig(
        output_dir="artifacts/ae-run",
        epochs=5,
        batch_size=256,
    ),
)

trainer.fit(loaders, metadata={"dataset": "glove", "model": "ae"})
```

Variational training:

```python
from autoencoders import VAETrainer, VariationalAutoencoderConfig, VariationalAutoencoderModel
from autoencoders.data.base import TensorSpec

trainer = VAETrainer(
    model=VariationalAutoencoderModel(
        config=VariationalAutoencoderConfig(
            latent_dim=16,
            kl_weight=0.1,
            free_bits=0.02,
            kl_warmup_epochs=20,
        ),
        sample_spec=TensorSpec(shape=(50,)),
        encoder="mlp",
        encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
        decoder="mlp",
        decoder_config={"hidden_dims": [64, 50], "activation": "relu", "use_bias": True},
    ),
    args=TrainingConfig(output_dir="artifacts/vae-run", epochs=10),
)
```

Quantized training:

```python
from autoencoders import VQTrainer, TrainingConfig, load_model
from autoencoders.data.base import TensorSpec

trainer = VQTrainer(
    model=load_model(
        "rqvae",
        sample_spec=TensorSpec(shape=(None, 50)),
        latent_dim=16,
        codebook_size=256,
        num_quantizers=4,
        use_ema_codebook=True,
        dead_code_reset=True,
        encoder="mlp",
        encoder_config={"hidden_dims": [64, 32], "activation": "relu", "use_bias": True},
        decoder="mlp",
        decoder_config={"hidden_dims": [64, 50], "activation": "relu", "use_bias": True},
    ),
    args=TrainingConfig(output_dir="artifacts/rqvae-run", epochs=10),
)
```

## Training Entry Point

Source checkouts now use one unified YAML-driven entrypoint:

- `examples/trainer.py`

The legacy `examples/train_ae.py` wrapper still forwards into the same code path for basic AE runs.

Useful examples:

```bash
python examples/trainer.py --config examples/configs/glove/ae.yaml --epoch 5
python examples/trainer.py --config examples/configs/glove/vae.yaml --epoch 5
python examples/trainer.py --config examples/configs/glove/vqvae.yaml --epoch 5
python examples/trainer.py --config examples/configs/cifar10/vqvae.yaml --epoch 5
python examples/trainer.py --config examples/configs/cifar10/vqvae_vit.yaml --epoch 5
```

Each config is organized into five sections:

- `dataset`
- `model`
- `encoder`
- `decoder`
- `trainer`

Each section uses `name + config` form except `trainer`, which is a flat config block. Runtime overrides such as `--epoch 5`, `--lr 0.001`, or `--max_vectors 5000` resolve into `${...:default}$` placeholders inside the YAML files before training starts.

## Launch-Ready Features

- `🗃️ Checkpoints`: `save_pretrained()` and `from_pretrained()`
- `📤 Exports`: standardized latent artifact export across model families
- `📚 Real datasets`: static embedding tables, sentence corpora, and CLIP-backed image-text corpora
- `🎛️ Family-specific trainers`: deterministic, variational, quantized, and adversarial flows
- `🧪 Packaging`: buildable `sdist` and wheel, ready for PyPI publication

## Design Direction

The library is organized around latent model families rather than a single monolithic interface:

- `BaseAutoencoderModel`
- `BaseVariationalAutoencoderModel`
- `BaseVectorQuantizedAutoencoderModel`

Matching outputs are also family-specific:

- `BaseAutoencoderOutput`
- `VariationalAutoencoderOutput`
- `QuantizedAutoencoderOutput`

This keeps the shared API stable without flattening away meaningful model differences such as posterior statistics or codebook indices.

## Current Scope

`autoencoders` is intentionally embedding-first, with a growing image path for CNN-backed quantized models. The current core is aimed at:

- representation learning on embedding matrices
- latent compression
- variational latent modeling
- quantized latent tokenization

Future raw-modality frontends and multimodal adapters can be layered on top of this core.

## Repository Status

This project is still early, but the current package already supports:

- trainable deterministic, variational, and quantized autoencoder families
- reusable checkpoints
- exportable latent artifacts
- real embedding datasets with download and cache support
- package metadata and distribution artifacts ready for publication workflows

## Development

Build the package locally:

```bash
python -m build
```

Check the generated distribution:

```bash
twine check dist/*
```
