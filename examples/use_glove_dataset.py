"""Minimal dataset usage example for training and evaluation loops."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders.data import load_dataset


def main() -> None:
    dataset = load_dataset("glove", dim=50, max_vectors=10000)
    loaders = dataset.get_dataloaders(batch_size=128)

    train_batch = next(iter(loaders.train))
    validation_batch = next(iter(loaders.validation))
    test_batch = next(iter(loaders.test))

    print("train batch:", tuple(train_batch.shape))
    print("validation batch:", tuple(validation_batch.shape))
    print("test batch:", tuple(test_batch.shape))


if __name__ == "__main__":
    main()
