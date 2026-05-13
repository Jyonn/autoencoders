# Datasets

## Recommended First Real Embedding Matrix

For the first real training and evaluation target, this project recommends:

**Stanford GloVe `glove.6B.50d`**

Why this choice works well:

- classic and widely recognized
- real pretrained embeddings, not synthetic toy data
- simple text format that matches the current `AutoencoderModel`
- small enough to experiment with compared to larger 300d or multi-million-vocabulary matrices

The official Stanford GloVe page lists the classic 2014 Wikipedia + Gigaword 5 release as:

- `6B` tokens
- `400K` uncased vocabulary
- `50d`, `100d`, `200d`, and `300d` variants
- distributed as `glove.6B.zip`

Official source:

- [Stanford GloVe project page](https://nlp.stanford.edu/projects/glove/)

## What Exists In The Library Now

The repository now includes a reusable dataset layer:

- [autoencoders/data/base.py](/Users/jyonn/Projects/Libraries/autoencoders/autoencoders/data/base.py): base dataset contracts, cache layout, deterministic splitting, and dataloader helpers
- [autoencoders/data/glove.py](/Users/jyonn/Projects/Libraries/autoencoders/autoencoders/data/glove.py): a concrete `GloVeDataset` with download, cache, prepare, split, and dataloader support

The intended usage is:

```python
from autoencoders.data import load_dataset

dataset = load_dataset("glove", dim=50, max_vectors=50000)
loaders = dataset.get_dataloaders(batch_size=256)
```

This gives a very direct path into training and evaluation:

- `loaders.train`
- `loaders.validation`
- `loaders.test`

By default, cached files live under:

```text
~/.cache/autoencoders
```

You can override that globally with:

```bash
export AUTOENCODERS_CACHE=/your/cache/path
```

The first call will automatically:

- download `glove.6B.zip` from Stanford
- extract `glove.6B.50d.txt`
- convert it into a torch-friendly artifact with `embeddings.pt`, `tokens.txt`, and `metadata.json`
- cache everything under the global autoencoders cache

## Train The Basic AE

The training example follows the same pattern and does not require an explicit prepare step:

```bash
/Users/jyonn/Projects/venv/library/bin/python examples/train_autoencoder.py --dataset glove --model ae --dim 50 --max-vectors 50000
```

For the denoising autoencoder:

```bash
/Users/jyonn/Projects/venv/library/bin/python examples/train_autoencoder.py --dataset glove --model dae --dim 50 --max-vectors 50000
```

For the variational autoencoder:

```bash
/Users/jyonn/Projects/venv/library/bin/python examples/train_autoencoder.py --dataset glove --model vae --dim 50 --max-vectors 50000
```

The VAE training defaults are tuned to avoid immediate collapse:

- `--kl-weight 0.1`
- `--kl-warmup-epochs 20`
- `--free-bits 0.02`

If you only want to inspect the dataset from Python, the direct usage stays simple:

```python
from autoencoders.data import load_dataset

dataset = load_dataset("glove", dim=50, max_vectors=10000)
loaders = dataset.get_dataloaders(batch_size=128)

train_batch = next(iter(loaders.train))
validation_batch = next(iter(loaders.validation))
test_batch = next(iter(loaders.test))
```

## Other Good Candidates

Besides GloVe, the next most reasonable datasets for this library are:

- `fastText English vectors`: strong general-purpose word embeddings, large coverage, but heavier than `glove.6B.50d`
- `GloVe 840B 300d`: still classic and real, but much larger and better treated as a later-stage benchmark
- `word2vec Google News 300d`: historically important, but distributed in a less convenient binary format
- `ConceptNet Numberbatch`: useful if we want semantically enriched word embeddings rather than purely distributional ones
- derived encoder features such as CLIP text or image embeddings: especially good once we start validating multimodal autoencoders

## How To Choose

- For the first deterministic AE: use `glove.6B.50d`
- For a stronger language embedding benchmark: use `fastText`
- For multimodal latent testing: use derived CLIP or SigLIP embeddings from a fixed corpus
- For later large-scale stress tests: use `GloVe 300d` or `Google News word2vec`

## Notes

- For quick iteration, keep `--max-vectors` small at first, such as `10000` or `50000`.
- The full classic matrix is useful for real training, but a capped subset is usually enough for architecture smoke tests.
- Once the pipeline is stable, we can add alternative real matrices such as `fastText`, `word2vec`, or CLIP text/image embeddings.
