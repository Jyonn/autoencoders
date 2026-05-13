"""Trainer abstractions for autoencoder-family models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..data.base import DatasetLoaders


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-facing device string into a torch device."""

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    """Set the global torch seed used by training."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingArguments:
    """High-level settings for autoencoder training runs."""

    output_dir: str = "artifacts/train-autoencoder"
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 256
    device: str = "auto"
    seed: int = 42


class AutoencoderTrainer:
    """A small trainer for autoencoder-family models."""

    def __init__(
        self,
        model,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.args = args
        self.device = resolve_device(args.device)
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
        )

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs=batch)
            outputs.loss.backward()
            self.optimizer.step()

            batch_size = batch.shape[0]
            total_loss += outputs.loss.detach().item() * batch_size
            total_examples += batch_size

        return total_loss / max(total_examples, 1)

    def evaluate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                outputs = self.model(inputs=batch)
                batch_size = batch.shape[0]
                total_loss += outputs.loss.detach().item() * batch_size
                total_examples += batch_size

        return total_loss / max(total_examples, 1)

    def fit(
        self,
        dataloaders: DatasetLoaders,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        best_validation_loss = float("inf")
        history: list[dict[str, float | int]] = []
        output_dir = Path(self.args.output_dir)

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(dataloaders.train)
            validation_loss = self.evaluate(dataloaders.validation)
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
                self.model.save_pretrained(output_dir / "best")

        test_loss = self.evaluate(dataloaders.test)
        self.model.save_pretrained(output_dir / "final")

        metrics = {
            "device": str(self.device),
            "best_validation_loss": best_validation_loss,
            "final_test_loss": test_loss,
            "history": history,
            "training_args": asdict(self.args),
        }
        if metadata:
            metrics.update(metadata)

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"final_test_loss={test_loss:.6f}")
        print(f"Saved final model to {output_dir / 'final'}")
        print(f"Saved best model to {output_dir / 'best'}")
        print(f"Saved metrics to {metrics_path}")
        return metrics

