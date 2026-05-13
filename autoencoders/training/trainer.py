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

    def train_epoch(self, dataloader) -> dict[str, float]:
        self.model.train()
        return self._run_epoch(dataloader, training=True)

    def evaluate(self, dataloader) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(dataloader, training=False)

    def fit(
        self,
        dataloaders: DatasetLoaders,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        best_validation_loss = float("inf")
        history: list[dict[str, float | int]] = []
        output_dir = Path(self.args.output_dir)

        for epoch in range(self.args.epochs):
            train_metrics = self.train_epoch(dataloaders.train)
            validation_metrics = self.evaluate(dataloaders.validation)
            epoch_metrics: dict[str, float | int] = {"epoch": epoch + 1}
            epoch_metrics.update({f"train_{name}": value for name, value in train_metrics.items()})
            epoch_metrics.update({f"validation_{name}": value for name, value in validation_metrics.items()})
            history.append(epoch_metrics)
            print(self._format_epoch_metrics(epoch_metrics))

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                self.model.save_pretrained(output_dir / "best")

        test_metrics = self.evaluate(dataloaders.test)
        self.model.save_pretrained(output_dir / "final")

        metrics = {
            "device": str(self.device),
            "best_validation_loss": best_validation_loss,
            "final_test_loss": test_metrics["loss"],
            "final_test_metrics": test_metrics,
            "history": history,
            "training_args": asdict(self.args),
        }
        if metadata:
            metrics.update(metadata)

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(self._format_metric_line("final_test", test_metrics))
        print(f"Saved final model to {output_dir / 'final'}")
        print(f"Saved best model to {output_dir / 'best'}")
        print(f"Saved metrics to {metrics_path}")
        return metrics

    def _run_epoch(self, dataloader, *, training: bool) -> dict[str, float]:
        totals: dict[str, float] = {}
        total_examples = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            if training:
                self.optimizer.zero_grad()
            outputs = self.model(inputs=batch)
            if training:
                outputs.loss.backward()
                self.optimizer.step()

            batch_size = batch.shape[0]
            total_examples += batch_size
            batch_metrics = self._extract_batch_metrics(outputs)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size

        return {name: total / max(total_examples, 1) for name, total in totals.items()}

    @staticmethod
    def _extract_batch_metrics(outputs) -> dict[str, float]:
        metrics: dict[str, float] = {"loss": float(outputs.loss.detach().item())}
        for name, value in outputs.loss_dict.items():
            if name == "loss":
                metrics["loss"] = float(value.detach().item()) if hasattr(value, "detach") else float(value)
            else:
                metrics[name] = float(value.detach().item()) if hasattr(value, "detach") else float(value)
        return metrics

    @staticmethod
    def _format_epoch_metrics(epoch_metrics: dict[str, float | int]) -> str:
        parts = [f"epoch={epoch_metrics['epoch']}"]
        for name, value in epoch_metrics.items():
            if name == "epoch":
                continue
            parts.append(f"{name}={float(value):.6f}")
        return " ".join(parts)

    @staticmethod
    def _format_metric_line(prefix: str, metrics: dict[str, float]) -> str:
        parts = [f"{prefix}_{name}={value:.6f}" for name, value in metrics.items()]
        return " ".join(parts)
