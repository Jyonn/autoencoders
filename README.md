# autoencoders

`autoencoders` is a unified library for building, training, and serving autoencoder-family models across continuous, variational, quantized, and masked latent spaces.

The goal is simple: make autoencoders feel as composable and reusable as `transformers`.

## Mission

Build the standard interface layer for latent models.

## Positioning

`autoencoders` is not just a collection of reconstruction models. It is a foundation for latent representation learning.

We aim to provide:

- A unified interface for `AE`, `VAE`, `VQ-VAE`, `AutoencoderKL`, `MAE`, and future multimodal autoencoder variants
- A consistent `Config + Model + Output + Processor + Pipeline` developer experience
- A shared abstraction over continuous, probabilistic, discrete, and masked latent spaces
- Pretrained model loading and reusable checkpoints through `from_pretrained()` and `save_pretrained()`

## Design Principles

### 1. One interface, many latent spaces

Different autoencoder families should feel native inside one library, even when their latent variables behave differently.

### 2. Latent-first, modality-aware

The core library should model latents and feature spaces cleanly, while modality-specific frontends handle raw inputs such as images, text, audio, video, and multimodal dictionaries.

### 3. Research-friendly, production-usable

Models should be easy to prototype, train, serialize, share, and serve without turning the library into a paper zoo.

## What We Are Not

- Not only a VAE benchmark repo
- Not only a diffusion-side image VAE toolkit
- Not only a learned compression library

## Model Families We Want To Support

- Vanilla and regularized autoencoders
- Variational autoencoders
- Vector-quantized autoencoders
- Masked autoencoders
- Latent autoencoders used by diffusion systems
- Multimodal latent autoencoders

## Library Shape

The long-term shape of the project should feel familiar:

```python
from autoencoders import AutoencoderModel

model = AutoencoderModel.from_pretrained("org/model-name")
outputs = model(inputs)

latents = outputs.latents
reconstruction = outputs.reconstruction
```

With task-friendly layers on top:

- `Processor` for raw modality inputs
- `Pipeline` for reconstruction, encoding, sampling, and compression-like workflows
- Structured outputs for reconstruction, posterior parameters, codebook indices, masks, and losses

## First Milestone

The first usable version should likely focus on:

- `AutoencoderModel`
- `VariationalAutoencoderModel`
- `VectorQuantizedAutoencoderModel`
- `AutoencoderKLModel`
- `MaskedAutoencoderModel`

Along with:

- `BaseAutoencoderConfig`
- `BaseAutoencoderModel`
- `PreTrainedAutoencoderModel`
- `AutoencoderOutput`
- `AutoencoderProcessor`
- `AutoencoderPipeline`

## Why Now

The current ecosystem is fragmented:

- Research-oriented VAE libraries exist
- Diffusion libraries include some latent autoencoders
- Compression libraries focus on codecs
- MAE-style models live elsewhere

But there is still no widely adopted standard library for the full autoencoder family.

## Status

This repository is at the very beginning. The immediate next step is to turn this positioning into a minimal but stable core architecture.

## Real Data

The first recommended real embedding matrix for training and testing is **Stanford GloVe `glove.6B.50d`**.

Downloadable datasets use a global cache by default at `~/.cache/autoencoders`, and you can override it with `AUTOENCODERS_CACHE`.

- Dataset notes: [docs/datasets.md](/Users/jyonn/Projects/Libraries/autoencoders/docs/datasets.md)
- Generic training example: [examples/train_autoencoder.py](/Users/jyonn/Projects/Libraries/autoencoders/examples/train_autoencoder.py)

The training flow is now also exposed as library primitives:

- `TrainingArguments`
- `AutoencoderTrainer`
