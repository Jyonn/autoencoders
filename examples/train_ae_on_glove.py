"""Minimal training example for the basic autoencoder on a real embedding matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders import (
    AutoencoderConfig,
    AutoencoderModel,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/ae-glove-50d", help="Model output directory.")
    parser.add_argument("--dim", type=int, default=50, help="GloVe embedding dimension.")
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=None,
        help="Optional cap for faster experiments.",
    )
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Encoder hidden dims.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("glove", dim=args.dim, max_vectors=args.max_vectors)
    dataloaders = dataset.get_dataloaders(batch_size=args.batch_size)
    dataloader = dataloaders.train
    embedding_matrix = dataset.load_embedding_matrix()

    config = AutoencoderConfig(
        input_dim=embedding_matrix.embedding_dim,
        latent_dim=args.latent_dim,
        hidden_dims=list(args.hidden_dims),
    )
    model = AutoencoderModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_examples = 0

        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs=batch)
            outputs.loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            total_loss += outputs.loss.detach().item() * batch_size
            total_examples += batch_size

        mean_loss = total_loss / max(total_examples, 1)
        print(f"epoch={epoch + 1} mean_loss={mean_loss:.6f}")

    output_dir = Path(args.output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()
