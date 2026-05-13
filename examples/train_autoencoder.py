"""Train and evaluate an autoencoder-family model on a named dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders import load_dataset, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="glove", help="Dataset name.")
    parser.add_argument("--model", default="ae", help="Model name, for example 'ae' or 'dae'.")
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

    parser.add_argument("--noise-type", default="gaussian", help="DAE noise type.")
    parser.add_argument("--noise-std", type=float, default=0.1, help="DAE gaussian noise std.")
    parser.add_argument("--masking-ratio", type=float, default=0.3, help="DAE masking ratio.")
    parser.add_argument(
        "--apply-noise-in-eval",
        action="store_true",
        help="Whether denoising autoencoders should also corrupt inputs during evaluation.",
    )

    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, mps.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if args.model.strip().lower() in {"dae", "denoising_autoencoder", "denoising-autoencoder"}:
        model_kwargs.update(
            {
                "noise_type": args.noise_type,
                "noise_std": args.noise_std,
                "masking_ratio": args.masking_ratio,
                "apply_noise_in_eval": args.apply_noise_in_eval,
            }
        )

    return load_model(args.model, **model_kwargs)


def evaluate(model, dataloader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(inputs=batch)
            batch_size = batch.shape[0]
            total_loss += outputs.loss.detach().item() * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset = build_dataset(args)
    dataloaders = dataset.get_dataloaders(
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    embedding_matrix = dataset.load_embedding_matrix()
    model = build_model(args, input_dim=embedding_matrix.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in dataloaders.train:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs=batch)
            outputs.loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            total_loss += outputs.loss.detach().item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / max(total_examples, 1)
        validation_loss = evaluate(model, dataloaders.validation, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save_pretrained(Path(args.output_dir) / "best")

    test_loss = evaluate(model, dataloaders.test, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "final")

    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "device": str(device),
        "best_validation_loss": best_validation_loss,
        "final_test_loss": test_loss,
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"final_test_loss={test_loss:.6f}")
    print(f"Saved final model to {output_dir / 'final'}")
    print(f"Saved best model to {output_dir / 'best'}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
