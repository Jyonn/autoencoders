# Positioning

## One-Sentence Mission

Build the standard interface layer for latent models.

## Product Thesis

`autoencoders` should be the library developers reach for when they want a clean, reusable abstraction for autoencoder-family models, in the same way they reach for `transformers` when they want a standard interface for sequence and multimodal foundation models.

The key distinction is that we are not organizing the project around papers. We are organizing it around latent-space operations:

- encode
- transform latent state
- decode
- reconstruct
- sample
- tokenize or quantize

## Core Positioning

The library should be framed as:

**A unified library for latent representation learning across continuous, variational, quantized, and masked latent spaces.**

This gives the project a larger and more durable scope than any one subcategory:

- broader than VAE implementations
- broader than diffusion VAEs
- broader than learned compression
- broader than masked pretraining for vision

## What The Library Must Unify

At minimum, the public API should eventually unify:

- configs
- model construction
- pretrained checkpoints
- encode and decode entrypoints
- standardized outputs
- processors for raw modality inputs
- pipelines for end-to-end tasks

## What Should Stay Flexible

The library should not force all models into one identical latent schema.

Examples:

- `VAE` models have posterior statistics and KL terms
- `VQ-VAE` models have quantized states and codebook indices
- `MAE` models have masks and reconstruction over masked regions
- latent diffusion autoencoders may expose codec-specific scaling behavior

The shared API should be stable, while model-specific fields remain extensible.

## Recommended North Star

If someone asks what this project is in one sentence, a good answer is:

**`transformers` for autoencoder-family models.**

If someone asks what problem it solves, a good answer is:

**It makes latent models interoperable.**

## Non-Goals

To avoid drift, these should not be the first identity of the project:

- a benchmark-only model zoo
- a paper reproduction collection
- a compression-only framework
- a diffusion-only helper package

## Initial Audience

The first users are likely to be:

- researchers prototyping representation learning methods
- engineers building latent codecs for generation systems
- teams that want reusable pretrained autoencoders and stable inference APIs
- multimodal practitioners who want one abstraction across modalities

## Phase 1 Model Scope

A strong first wave would include:

- `AutoencoderModel`
- `VariationalAutoencoderModel`
- `BetaVAEModel`
- `VectorQuantizedAutoencoderModel`
- `AutoencoderKLModel`
- `MaskedAutoencoderModel`

This is enough to establish the abstraction without overextending the implementation surface.

## Architectural Consequence

This positioning implies a specific design decision:

The core model layer should prefer `features` or modality-adapted tensors as the canonical input, while raw images, text tokens, waveforms, and multimodal bundles are handled by processors and frontend modules.

That keeps the library centered on latent modeling instead of raw input preprocessing.

## Immediate Next Steps

1. Define base configs, outputs, and model interfaces.
2. Decide the canonical `forward`, `encode`, and `decode` signatures.
3. Implement one model from each major latent family.
4. Add a small pretrained checkpoint and round-trip save/load tests.
5. Introduce processors and pipelines only after the model contract is stable.

