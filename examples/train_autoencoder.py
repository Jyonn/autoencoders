"""Train and evaluate an autoencoder-family model on a named dataset."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders import (
    AutoencoderTrainer,
    QuantizedAutoencoderTrainer,
    QuantizedAutoencoderTrainingArguments,
    TrainingArguments,
    VAETrainer,
    VAETrainingArguments,
    load_dataset,
    load_model,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="glove", choices=["glove"], help="Dataset name.")
    parser.add_argument("--model", default="ae", choices=["ae", "dae", "sae", "vae", "betavae", "vqvae"], help="Model name.")
    parser.add_argument("--output-dir", default="artifacts/train-autoencoder", help="Model output directory.")

    parser.add_argument("--dim", type=int, default=50, help="Dataset embedding dimension when supported.")
    parser.add_argument("--max-vectors", type=int, default=None, help="Optional dataset cap for faster experiments.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting and training.")

    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Encoder hidden dims.")
    parser.add_argument("--activation", default="relu", help="Activation name.")
    parser.add_argument("--reconstruction-loss", default="mse", help="Reconstruction loss name.")
    parser.add_argument("--sparsity-weight", type=float, default=1e-3, help="Sparse-AE latent L1 regularization weight.")
    parser.add_argument("--kl-weight", type=float, default=0.1, help="VAE KL loss weight.")
    parser.add_argument("--beta", type=float, default=4.0, help="Beta-VAE KL multiplier.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=20, help="Number of epochs for VAE KL warmup.")
    parser.add_argument("--kl-start-weight", type=float, default=0.0, help="Starting KL weight during warmup.")
    parser.add_argument("--free-bits", type=float, default=0.02, help="Per-latent-dimension free bits floor for VAE KL.")
    parser.add_argument("--codebook-size", type=int, default=256, help="VQ-VAE codebook size.")
    parser.add_argument("--commitment-weight", type=float, default=0.25, help="VQ-VAE commitment loss weight.")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="VQ-VAE codebook loss weight.")

    parser.add_argument("--noise-type", default="gaussian", help="DAE noise type.")
    parser.add_argument("--noise-std", type=float, default=0.1, help="DAE gaussian noise std.")
    parser.add_argument("--masking-ratio", type=float, default=0.3, help="DAE masking ratio.")
    parser.add_argument(
        "--apply-noise-in-eval",
        action="store_true",
        help="Whether denoising autoencoders should also corrupt inputs during evaluation.",
    )

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
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    dataset_kwargs = {
        "dim": args.dim,
        "max_vectors": args.max_vectors,
    }
    return load_dataset(args.dataset, **dataset_kwargs)


def build_model(args: argparse.Namespace, input_dim: int):
    model_kwargs = {
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "reconstruction_loss": args.reconstruction_loss,
    }

    if args.model == "dae":
        model_kwargs.update(
            {
                "noise_type": args.noise_type,
                "noise_std": args.noise_std,
                "masking_ratio": args.masking_ratio,
                "apply_noise_in_eval": args.apply_noise_in_eval,
            }
        )
    if args.model == "sae":
        model_kwargs.update(
            {
                "sparsity_weight": args.sparsity_weight,
            }
        )
    if args.model == "vae":
        model_kwargs.update(
            {
                "kl_weight": args.kl_weight,
            }
        )
    if args.model == "betavae":
        model_kwargs.update(
            {
                "beta": args.beta,
            }
        )
    if args.model == "vqvae":
        model_kwargs.update(
            {
                "codebook_size": args.codebook_size,
                "commitment_weight": args.commitment_weight,
                "codebook_weight": args.codebook_weight,
            }
        )

    return load_model(args.model, **model_kwargs)


def build_trainer(args: argparse.Namespace, model):
    common_kwargs = {
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "show_only_best_epochs": args.show_only_best_epochs,
    }

    if args.model == "vae" or args.model == "betavae":
        training_args = VAETrainingArguments(
            kl_warmup_epochs=args.kl_warmup_epochs,
            kl_start_weight=args.kl_start_weight,
            free_bits=args.free_bits,
            **common_kwargs,
        )
        return VAETrainer(model=model, args=training_args)
    if args.model == "vqvae":
        training_args = QuantizedAutoencoderTrainingArguments(**common_kwargs)
        return QuantizedAutoencoderTrainer(model=model, args=training_args)

    training_args = TrainingArguments(**common_kwargs)
    return AutoencoderTrainer(model=model, args=training_args)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = build_dataset(args)
    dataloaders = dataset.get_dataloaders(
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    embedding_matrix = dataset.load_embedding_matrix()
    model = build_model(args, input_dim=embedding_matrix.embedding_dim)
    trainer = build_trainer(args, model)
    trainer.fit(
        dataloaders,
        metadata={
            "dataset": args.dataset,
            "model": args.model,
        },
    )



if __name__ == "__main__":
    main()
