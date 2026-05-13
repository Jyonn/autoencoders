"""Download and preprocess the classic GloVe 6B 50d embedding matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders.data import GloVeDataset

DEFAULT_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Source archive URL.")
    parser.add_argument(
        "--dim",
        type=int,
        default=50,
        help="Embedding dimension to extract.",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=None,
        help="Optional cap for faster experiments.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing cached archive instead of downloading again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = GloVeDataset(dim=args.dim, max_vectors=args.max_vectors)
    if args.url != DEFAULT_URL:
        dataset.base_url = args.url

    artifact_dir = dataset.ensure_prepared(
        download=not args.skip_download,
        force_download=False,
        force_prepare=False,
    )
    embedding_matrix = dataset.load_embedding_matrix(download=False)

    print(f"Prepared {embedding_matrix.num_embeddings} embeddings with dim {embedding_matrix.embedding_dim}")
    print(f"Saved artifact to {artifact_dir}")


if __name__ == "__main__":
    main()
