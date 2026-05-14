"""Shared helpers for training entrypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders import TrainingArguments, load_dataset, set_seed


DATASET_CHOICES = ["glove", "fasttext", "numberbatch"]


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="glove", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument("--output-dir", default="artifacts/train-autoencoder", help="Model output directory.")
    parser.add_argument("--dim", type=int, default=50, help="Dataset embedding dimension when supported.")
    parser.add_argument("--max-vectors", type=int, default=None, help="Optional dataset cap for faster experiments.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting and training.")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs. Set to 0 to train until early stopping triggers.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs without validation improvement.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, mps.")
    parser.add_argument(
        "--show-only-best-epochs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to persist only best-epoch summaries in the terminal output.",
    )


def build_dataset(args: argparse.Namespace):
    dim = args.dim
    if args.dataset in {"fasttext", "numberbatch"} and dim == 50:
        dim = 300
    return load_dataset(
        args.dataset,
        dim=dim,
        max_vectors=args.max_vectors,
    )


def prepare_training(args: argparse.Namespace):
    set_seed(args.seed)
    dataset = build_dataset(args)
    dataloaders = dataset.get_dataloaders(
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    embedding_matrix = dataset.load_embedding_matrix()
    return dataset, dataloaders, embedding_matrix.embedding_dim


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        show_only_best_epochs=args.show_only_best_epochs,
    )
