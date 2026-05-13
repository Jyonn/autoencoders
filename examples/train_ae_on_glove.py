"""Compatibility wrapper for training a basic autoencoder on GloVe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_autoencoder import main as train_autoencoder_main


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
    sys.argv = [
        sys.argv[0],
        "--dataset",
        "glove",
        "--model",
        "ae",
        "--output-dir",
        args.output_dir,
        "--dim",
        str(args.dim),
        "--latent-dim",
        str(args.latent_dim),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
    ]
    if args.max_vectors is not None:
        sys.argv.extend(["--max-vectors", str(args.max_vectors)])
    if args.hidden_dims:
        sys.argv.append("--hidden-dims")
        sys.argv.extend(str(dim) for dim in args.hidden_dims)
    train_autoencoder_main()


if __name__ == "__main__":
    main()
