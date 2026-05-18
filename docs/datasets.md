# Datasets

## Current Dataset Surface

The repository now ships with a dataset layer that mirrors the model/module architecture:

- `autoencoders/data/base.py`: base contracts, caching, deterministic splits, and `DataSpec`
- `autoencoders/data/glove.py`: downloadable GloVe embeddings
- `autoencoders/data/fasttext.py`: official fastText English vectors
- `autoencoders/data/numberbatch.py`: ConceptNet Numberbatch vectors
- `autoencoders/data/text.py`: shared infrastructure for encoder-backed text datasets
- `autoencoders/data/snli.py`: SNLI embeddings
- `autoencoders/data/multinli.py`: MultiNLI embeddings
- `autoencoders/data/clip.py`: shared CLIP-backed multimodal infrastructure
- `autoencoders/data/flickr30k.py`: Flickr30k CLIP embeddings
- `autoencoders/data/cifar10.py`: CIFAR-10 image tensors for CNN- and ViT-backed experiments

## Recommended Starting Points

- For embedding-first deterministic and variational experiments: `glove.6B.50d`
- For stronger word-level coverage: `fasttext`
- For semantically enriched embeddings: `numberbatch`
- For sentence-level latent experiments: `snli` or `multinli`
- For image-text representation experiments: `flickr30k`
- For image-backed quantized or transformer experiments: `cifar10`

## Python Usage

Load a dataset directly:

```python
from autoencoders.data import load_dataset

dataset = load_dataset("glove", dim=50, max_vectors=50000)
loaders = dataset.get_dataloaders(batch_size=256)
print(dataset.get_sample_spec())  # TensorSpec(shape=(50,))
```

Sentence datasets materialize embeddings through a configured encoder:

```python
dataset = load_dataset(
    "snli",
    encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    max_examples=50000,
)
print(dataset.get_sample_spec())
```

Image datasets expose `H x W x C` specs:

```python
dataset = load_dataset("cifar10", max_examples=10000)
print(dataset.get_sample_spec())  # TensorSpec(shape=(32, 32, 3))
```

## YAML Training Flow

Training now goes through one YAML-first entrypoint:

```bash
python examples/trainer.py --config examples/configs/glove/ae.yaml --epoch 5
python examples/trainer.py --config examples/configs/glove/vae.yaml --epoch 5
python examples/trainer.py --config examples/configs/glove/vqvae.yaml --epoch 5
python examples/trainer.py --config examples/configs/cifar10/vqvae.yaml --epoch 5
python examples/trainer.py --config examples/configs/cifar10/vqvae_vit.yaml --epoch 5
```

Each config is structured as:

- `dataset.name`
- `dataset.config`
- `model.name`
- `model.config`
- `encoder.name`
- `encoder.config`
- `decoder.name`
- `decoder.config`
- `trainer`

`decoder` may be `null`, but only when reversing the encoder yields a decoder whose runtime input spec matches the model's decoder input spec. Hierarchical and latent-shape-changing models should declare an explicit decoder.

## Caching

Downloaded datasets are cached under:

```text
~/.cache/autoencoders
```

Override globally with:

```bash
export AUTOENCODERS_CACHE=/your/cache/path
```

The first preparation step downloads raw assets, converts them into torch-friendly artifacts, and reuses those artifacts on later runs.

## Notes

- Keep `max_vectors` or `max_examples` modest for quick smoke tests.
- Encoder-backed text datasets usually take longer on the first run because they materialize embeddings before training starts.
- `cifar10` now retries mirrored downloads and validates cached archives before extraction, which makes interrupted downloads easier to recover from.
- Explicit image decoders should set `transpose: true` when they are intended to upsample back to image space.
