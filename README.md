<div align="center">

# autoencoders

**A latent-model toolkit for deterministic, variational, and quantized autoencoders**

<p>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/model_families-16%2B-111827" alt="16+ model families" />
  <img src="https://img.shields.io/badge/datasets-glove%20%7C%20fasttext%20%7C%20numberbatch%20%7C%20snli%20%7C%20multinli-0F766E" alt="Datasets" />
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
- Variational models: `VAE`, `DVAE`, `BetaVAE`, `HVAE`
- Quantized models: `VQVAE`, `FSQ`, `PQVAE`, `RQVAE`

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

Install everything commonly needed for experiments:

```bash
pip install "autoencoders[all]"
```

If you are working from source and plan to build or publish packages:

```bash
pip install "autoencoders[dev]"
```

## Quick Start

Create a basic autoencoder:

```python
from autoencoders import AutoencoderConfig, AutoencoderModel

config = AutoencoderConfig(
    input_dim=50,
    latent_dim=16,
    hidden_dims=[128, 64],
)

model = AutoencoderModel(config)
```

Run a forward pass:

```python
import torch

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

Export model artifacts for downstream use:

```python
artifact = model.export(inputs)

print(artifact.latents.shape)
print(artifact.reconstruction.shape)
```

## Product Surface

Use the package at three different layers:

- `Model layer`: build or load latent models with typed configs
- `Training layer`: train deterministic, variational, or quantized families with dedicated trainers
- `Experiment layer`: run curated shell scripts with model-specific defaults on real datasets

## Model Loading

Load a model dynamically by name:

```python
from autoencoders import load_model

model = load_model(
    "vae",
    input_dim=50,
    latent_dim=16,
    hidden_dims=[128, 64],
    kl_weight=0.1,
    free_bits=0.02,
    kl_warmup_epochs=20,
)
```

## Datasets

The library currently ships with embedding-first datasets:

- `glove`
- `fasttext`
- `numberbatch`
- `snli`
- `multinli`

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

Downloaded datasets use a global cache:

- default: `~/.cache/autoencoders`
- override with: `AUTOENCODERS_CACHE=/your/cache/path`

This makes the package useful both as:

- a standalone training library
- a latent-model subsystem inside larger PyTorch projects

## Training API

Deterministic training:

```python
from autoencoders import AETrainer, TrainingArguments

trainer = AETrainer(
    model=model,
    args=TrainingArguments(
        output_dir="artifacts/ae-run",
        epochs=5,
        batch_size=256,
    ),
)

trainer.fit(loaders, metadata={"dataset": "glove", "model": "ae"})
```

Variational training:

```python
from autoencoders import VAETrainer

trainer = VAETrainer(
    model=load_model(
        "vae",
        input_dim=50,
        latent_dim=16,
        hidden_dims=[128, 64],
        kl_weight=0.1,
        free_bits=0.02,
        kl_warmup_epochs=20,
    ),
    args=TrainingArguments(output_dir="artifacts/vae-run", epochs=10),
)
```

Quantized training:

```python
from autoencoders import VQTrainer

trainer = VQTrainer(
    model=load_model(
        "rqvae",
        input_dim=50,
        latent_dim=16,
        hidden_dims=[128, 64],
        codebook_size=256,
        num_quantizers=4,
        use_ema_codebook=True,
        dead_code_reset=True,
    ),
    args=TrainingArguments(output_dir="artifacts/rqvae-run", epochs=10),
)
```

## Training Scripts

From a source checkout, there are family-specific entrypoints:

- `examples/train_ae.py`
- `examples/train_vae.py`
- `examples/train_vq.py`

There are also convenience shell wrappers for common dataset/model combinations:

- `scripts/train_glove_*.sh`
- `scripts/train_fasttext_ae.sh`
- `scripts/train_numberbatch_ae.sh`

Examples:

```bash
bash scripts/train_glove_ae.sh
bash scripts/train_glove_vae.sh
bash scripts/train_glove_rqvae.sh
bash scripts/train_fasttext_ae.sh
```

Each wrapper includes model-specific defaults and still accepts extra CLI overrides.

## Launch-Ready Features

- `🗃️ Checkpoints`: `save_pretrained()` and `from_pretrained()`
- `📤 Exports`: standardized latent artifact export across model families
- `📚 Real datasets`: static embedding tables plus encoder-backed sentence corpora
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

`autoencoders` is intentionally embedding-first right now. The current core is aimed at:

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
