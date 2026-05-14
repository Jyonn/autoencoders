# autoencoders

`autoencoders` is a PyTorch-first library for autoencoder-family models across deterministic, variational, and quantized latent spaces.

The project goal is simple: make autoencoders feel composable, serializable, and reusable in the same way `transformers` did for sequence models.

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

## Installation

Install the package:

```bash
pip install autoencoders
```

Install with PyTorch dependencies:

```bash
pip install "autoencoders[torch]"
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

Load a dataset directly:

```python
from autoencoders import load_dataset

dataset = load_dataset("glove", dim=50, max_vectors=50000)
loaders = dataset.get_dataloaders(batch_size=256)
```

Downloaded datasets use a global cache:

- default: `~/.cache/autoencoders`
- override with: `AUTOENCODERS_CACHE=/your/cache/path`

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

## Development

Build the package locally:

```bash
python -m build
```

Check the generated distribution:

```bash
twine check dist/*
```
