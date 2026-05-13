"""Trainer abstractions for autoencoder-family models."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from pigmento import Bracket, Color, Pigmento, Prefix

from ..data.base import DatasetLoaders


def _build_logger(label: str, color: Color, *, inline: bool = False) -> Pigmento:
    logger = Pigmento()
    logger.set_display_mode(display_method_name=False, display_class_name=False)
    logger.add_prefix(Prefix(label, bracket=Bracket.DEFAULT, color=color))
    if inline:
        def inline_printer(prefixes, prefix_s, prefix_s_with_color, text, **kwargs):
            end = kwargs.pop("end", "")
            sys.stdout.write(f"\r{prefix_s_with_color} {text}{end}")
            sys.stdout.flush()

        logger.set_basic_printer(inline_printer)
    return logger


RUN_LOGGER = _build_logger("RUN", Color.BLUE)
STEP_LOGGER = _build_logger("STEP", Color.CYAN, inline=True)
EPOCH_LOGGER = _build_logger("EPOCH", Color.GREEN)
FINAL_LOGGER = _build_logger("FINAL", Color.MAGENTA)
SAVE_LOGGER = _build_logger("SAVE", Color.YELLOW)


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
    patience: int | None = None
    learning_rate: float = 1e-3
    batch_size: int = 256
    device: str = "auto"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be greater than or equal to 0.")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("patience must be greater than 0 when provided.")
        if self.epochs == 0 and self.patience is None:
            raise ValueError("patience must be provided when epochs is set to 0.")


@dataclass
class VAETrainingArguments(TrainingArguments):
    """Training settings specific to variational autoencoders."""

    kl_warmup_epochs: int = 20
    kl_start_weight: float = 0.0
    free_bits: float = 0.02

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be greater than or equal to 0.")
        if self.kl_start_weight < 0:
            raise ValueError("kl_start_weight must be non-negative.")
        if self.free_bits < 0:
            raise ValueError("free_bits must be non-negative.")


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
        self.current_epoch = 0
        self.max_epochs: int | None = None
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
        best_epoch: int | None = None
        epochs_without_improvement = 0
        stopped_early = False
        history: list[dict[str, float | int]] = []
        output_dir = Path(self.args.output_dir)
        epoch = 0
        max_epochs = self.args.epochs if self.args.epochs > 0 else None
        self.max_epochs = max_epochs
        self._log_run_start(metadata=metadata, max_epochs=max_epochs)

        while max_epochs is None or epoch < max_epochs:
            epoch += 1
            self.on_epoch_start(epoch)
            train_metrics = self.train_epoch(dataloaders.train)
            validation_metrics = self.evaluate(dataloaders.validation)
            epoch_metrics: dict[str, float | int] = {"epoch": epoch}
            epoch_metrics.update(self.get_epoch_metrics())
            epoch_metrics.update({f"train_{name}": value for name, value in train_metrics.items()})
            epoch_metrics.update({f"validation_{name}": value for name, value in validation_metrics.items()})
            history.append(epoch_metrics)
            improved = validation_metrics["loss"] < best_validation_loss
            self._log_epoch_summary(epoch_metrics, best_validation_loss, improved)

            if improved:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                self.model.save_pretrained(output_dir / "best")
            else:
                epochs_without_improvement += 1
                if self.args.patience is not None and epochs_without_improvement >= self.args.patience:
                    stopped_early = True
                    break

        test_metrics = self.evaluate(dataloaders.test)
        self.model.save_pretrained(output_dir / "final")

        metrics = {
            "device": str(self.device),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "final_test_loss": test_metrics["loss"],
            "final_test_metrics": test_metrics,
            "history": history,
            "stopped_early": stopped_early,
            "training_args": asdict(self.args),
        }
        if metadata:
            metrics.update(metadata)

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        self._log_run_end(
            test_metrics=test_metrics,
            output_dir=output_dir,
            metrics_path=metrics_path,
            stopped_early=stopped_early,
            best_epoch=best_epoch,
        )
        return metrics

    def on_epoch_start(self, epoch: int) -> None:
        self.current_epoch = epoch

    def get_epoch_metrics(self) -> dict[str, float | int]:
        return {}

    def _run_epoch(self, dataloader, *, training: bool) -> dict[str, float]:
        totals: dict[str, float] = {}
        total_examples = 0
        total_batches = len(dataloader)

        for batch_index, batch in enumerate(dataloader, start=1):
            batch = batch.to(self.device)
            if training:
                self.optimizer.zero_grad()
            outputs = self.forward_batch(batch, training=training)
            loss = self.compute_batch_loss(outputs, training=training)
            if training:
                loss.backward()
                self.optimizer.step()

            batch_size = batch.shape[0]
            total_examples += batch_size
            batch_metrics = self._extract_batch_metrics(outputs, loss=loss, training=training)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            if training:
                running_metrics = {name: total / max(total_examples, 1) for name, total in totals.items()}
                self._log_batch_progress(batch_index, total_batches, running_metrics)

        if training and total_batches:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return {name: total / max(total_examples, 1) for name, total in totals.items()}

    def forward_batch(self, batch: torch.Tensor, *, training: bool):
        return self.model(inputs=batch)

    def compute_batch_loss(self, outputs, *, training: bool) -> torch.Tensor:
        return outputs.loss

    def _extract_batch_metrics(self, outputs, *, loss: torch.Tensor, training: bool) -> dict[str, float]:
        metrics: dict[str, float] = {"loss": float(loss.detach().item())}
        for name, value in outputs.loss_dict.items():
            if name == "loss":
                continue
            else:
                metrics[name] = float(value.detach().item()) if hasattr(value, "detach") else float(value)
        return metrics

    def _log_run_start(self, metadata: dict[str, Any] | None, max_epochs: int | None) -> None:
        model_name = metadata.get("model", self.model.__class__.__name__) if metadata else self.model.__class__.__name__
        dataset_name = metadata.get("dataset", "unknown") if metadata else "unknown"
        epoch_budget = "early-stop" if max_epochs is None else str(max_epochs)
        RUN_LOGGER(
            f"model={model_name} dataset={dataset_name} device={self.device} epochs={epoch_budget}"
        )

    def _log_batch_progress(self, batch_index: int, total_batches: int, metrics: dict[str, float]) -> None:
        progress_width = 18
        filled = int(progress_width * batch_index / max(total_batches, 1))
        bar = "█" * filled + "·" * (progress_width - filled)
        parts = [
            f"{batch_index:>3}/{total_batches:<3}",
            bar,
            f"loss {metrics['loss']:.4f}",
        ]
        if "reconstruction_loss" in metrics:
            parts.append(f"recon {metrics['reconstruction_loss']:.4f}")
        if "kl_loss" in metrics:
            parts.append(f"kl {metrics['kl_loss']:.4f}")
        STEP_LOGGER("  ".join(parts))

    def _log_epoch_summary(
        self,
        epoch_metrics: dict[str, float | int],
        best_validation_loss: float,
        improved: bool,
    ) -> None:
        epoch = int(epoch_metrics["epoch"])
        if self.max_epochs is None:
            epoch_label = f"{epoch:03d}"
        else:
            epoch_label = f"{epoch:03d}/{self.max_epochs:03d}"

        summary_parts = [
            f"{epoch_label}",
            f"train {float(epoch_metrics['train_loss']):.4f}",
            f"valid {float(epoch_metrics['validation_loss']):.4f}",
        ]

        if "train_reconstruction_loss" in epoch_metrics and "validation_reconstruction_loss" in epoch_metrics:
            summary_parts.append(
                f"recon {float(epoch_metrics['train_reconstruction_loss']):.4f}/{float(epoch_metrics['validation_reconstruction_loss']):.4f}"
            )
        if "train_kl_loss" in epoch_metrics and "validation_kl_loss" in epoch_metrics:
            summary_parts.append(
                f"kl {float(epoch_metrics['train_kl_loss']):.4f}/{float(epoch_metrics['validation_kl_loss']):.4f}"
            )
        if "kl_weight" in epoch_metrics:
            summary_parts.append(f"beta {float(epoch_metrics['kl_weight']):.3f}")
        if "free_bits" in epoch_metrics:
            summary_parts.append(f"free {float(epoch_metrics['free_bits']):.3f}")
        if "train_free_bits_kl_loss" in epoch_metrics and "validation_free_bits_kl_loss" in epoch_metrics:
            summary_parts.append(
                f"free-kl {float(epoch_metrics['train_free_bits_kl_loss']):.4f}/{float(epoch_metrics['validation_free_bits_kl_loss']):.4f}"
            )

        summary_parts.append("best" if improved else f"best {best_validation_loss:.4f}")
        EPOCH_LOGGER("  ".join(summary_parts))

    def _log_run_end(
        self,
        test_metrics: dict[str, float],
        output_dir: Path,
        metrics_path: Path,
        stopped_early: bool,
        best_epoch: int | None,
    ) -> None:
        summary_parts = [f"test {test_metrics['loss']:.4f}"]
        if "reconstruction_loss" in test_metrics:
            summary_parts.append(f"recon {test_metrics['reconstruction_loss']:.4f}")
        if "kl_loss" in test_metrics:
            summary_parts.append(f"kl {test_metrics['kl_loss']:.4f}")
        if "free_bits_kl_loss" in test_metrics:
            summary_parts.append(f"free-kl {test_metrics['free_bits_kl_loss']:.4f}")
        if stopped_early and best_epoch is not None:
            summary_parts.append(f"stopped@{self.current_epoch} best@{best_epoch}")
        FINAL_LOGGER("  ".join(summary_parts))
        SAVE_LOGGER(f"best -> {output_dir / 'best'}")
        SAVE_LOGGER(f"final -> {output_dir / 'final'}")
        SAVE_LOGGER(f"metrics -> {metrics_path}")


class VAETrainer(AutoencoderTrainer):
    """Trainer with VAE-specific objective scheduling hooks."""

    def __init__(
        self,
        model,
        args: VAETrainingArguments,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        super().__init__(model=model, args=args, optimizer=optimizer)

    @property
    def vae_args(self) -> VAETrainingArguments:
        return self.args

    def get_current_kl_weight(self) -> float:
        target_weight = float(self.model.config.kl_weight)
        warmup_epochs = self.vae_args.kl_warmup_epochs
        start_weight = self.vae_args.kl_start_weight

        if warmup_epochs <= 0:
            return target_weight
        if warmup_epochs == 1:
            return target_weight
        if self.current_epoch <= 1:
            return start_weight

        progress = min((self.current_epoch - 1) / (warmup_epochs - 1), 1.0)
        return start_weight + progress * (target_weight - start_weight)

    def get_epoch_metrics(self) -> dict[str, float | int]:
        return {
            "free_bits": self.vae_args.free_bits,
            "kl_weight": self.get_current_kl_weight(),
        }

    def compute_free_bits_kl_loss(self, outputs) -> torch.Tensor:
        if outputs.posterior_mean is None or outputs.posterior_logvar is None:
            raise ValueError("VAETrainer requires outputs with posterior statistics.")

        kl_per_dim = -0.5 * (
            1
            + outputs.posterior_logvar
            - outputs.posterior_mean.pow(2)
            - outputs.posterior_logvar.exp()
        )
        mean_kl_per_dim = kl_per_dim.mean(dim=0)
        return torch.clamp(mean_kl_per_dim, min=self.vae_args.free_bits).sum()

    def compute_batch_loss(self, outputs, *, training: bool) -> torch.Tensor:
        if outputs.reconstruction_loss is None or outputs.kl_loss is None:
            raise ValueError("VAETrainer requires outputs with reconstruction_loss and kl_loss.")
        effective_kl_loss = self.compute_free_bits_kl_loss(outputs)
        return outputs.reconstruction_loss + self.get_current_kl_weight() * effective_kl_loss

    def _extract_batch_metrics(self, outputs, *, loss: torch.Tensor, training: bool) -> dict[str, float]:
        metrics = super()._extract_batch_metrics(outputs, loss=loss, training=training)
        metrics["free_bits_kl_loss"] = float(self.compute_free_bits_kl_loss(outputs).detach().item())
        return metrics
