"""Trainer abstractions for autoencoder-family models."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..data.base import DatasetLoaders


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
FG = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
}
BG = {
    "black": "40",
    "red": "41",
    "green": "42",
    "yellow": "43",
    "blue": "44",
    "magenta": "45",
    "cyan": "46",
    "white": "47",
}


def _style(
    text: str,
    *,
    fg: str | None = None,
    bg: str | None = None,
    bold: bool = False,
    dim: bool = False,
) -> str:
    codes: list[str] = []
    if bold:
        codes.append(BOLD[2:-1])
    if dim:
        codes.append(DIM[2:-1])
    if fg is not None:
        codes.append(FG[fg])
    if bg is not None:
        codes.append(BG[bg])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}{RESET}"


def _visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def _format_label(name: str, *, fg: str, bg: str) -> str:
    return _style(f" {name} ", fg=fg, bg=bg, bold=True)


def _format_metric(name: str, value: str, *, value_fg: str = "white") -> str:
    return f"{_style(name, fg='cyan', dim=True)} {_style(value, fg=value_fg, bold=True)}"


def _join_segments(*segments: str) -> str:
    separator = _style(" • ", fg="magenta", dim=True)
    return separator.join(segment for segment in segments if segment)


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
    show_only_best_epochs: bool = True

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


@dataclass
class TrainerDisplayConfig:
    """Terminal display settings for trainer logs."""

    separator: str = " • "
    progress_width: int = 18
    progress_fill: str = "█"
    progress_empty: str = "·"
    run_label_fg: str = "white"
    run_label_bg: str = "blue"
    best_label_fg: str = "white"
    best_label_bg: str = "green"
    epoch_label_fg: str = "white"
    epoch_label_bg: str = "cyan"
    final_label_fg: str = "white"
    final_label_bg: str = "magenta"
    save_label_fg: str = "black"
    save_label_bg: str = "yellow"
    phase_fg: str = "white"
    phase_bg: str = "cyan"
    epoch_index_fg: str = "yellow"
    batch_index_fg: str = "blue"
    progress_fg: str = "green"
    metric_name_fg: str = "cyan"
    metric_value_fg: str = "white"
    recon_value_fg: str = "green"
    kl_value_fg: str = "magenta"
    sparse_value_fg: str = "yellow"
    free_kl_value_fg: str = "blue"
    meta_value_fg: str = "yellow"

    def __post_init__(self) -> None:
        if self.progress_width <= 0:
            raise ValueError("progress_width must be greater than 0.")
        for color_name in (
            self.run_label_fg,
            self.run_label_bg,
            self.best_label_fg,
            self.best_label_bg,
            self.epoch_label_fg,
            self.epoch_label_bg,
            self.final_label_fg,
            self.final_label_bg,
            self.save_label_fg,
            self.save_label_bg,
            self.phase_fg,
            self.phase_bg,
            self.epoch_index_fg,
            self.batch_index_fg,
            self.progress_fg,
            self.metric_name_fg,
            self.metric_value_fg,
            self.recon_value_fg,
            self.kl_value_fg,
            self.sparse_value_fg,
            self.free_kl_value_fg,
            self.meta_value_fg,
        ):
            if color_name not in FG and color_name not in BG:
                raise ValueError(f"Unsupported display color: {color_name}")


class AutoencoderTrainer:
    """A small trainer for autoencoder-family models."""

    def __init__(
        self,
        model,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer | None = None,
        display: TrainerDisplayConfig | None = None,
    ) -> None:
        self.model = model
        self.args = args
        self.display = display or TrainerDisplayConfig()
        self.current_epoch = 0
        self.max_epochs: int | None = None
        self._last_live_line_length = 0
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

            if improved:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                self.model.save_pretrained(output_dir / "best")
                self._clear_live_line()
                self._log_best_epoch(epoch_metrics)
            else:
                if self.args.show_only_best_epochs:
                    self._log_epoch_summary(epoch_metrics, persist=False)
                else:
                    self._clear_live_line()
                    self._log_epoch_summary(epoch_metrics, persist=True)
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
        self._print_log(
            "RUN",
            self._join_segments(
                self._format_metric("model", model_name, value_fg=self.display.metric_value_fg),
                self._format_metric("dataset", dataset_name, value_fg=self.display.metric_value_fg),
                self._format_metric("device", str(self.device), value_fg=self.display.meta_value_fg),
                self._format_metric("epochs", epoch_budget, value_fg=self.display.progress_fg),
            ),
            fg=self.display.run_label_fg,
            bg=self.display.run_label_bg,
        )

    def _log_batch_progress(self, batch_index: int, total_batches: int, metrics: dict[str, float]) -> None:
        progress_width = self.display.progress_width
        filled = int(progress_width * batch_index / max(total_batches, 1))
        bar = self.display.progress_fill * filled + self.display.progress_empty * (progress_width - filled)
        parts = [
            _style(self._format_epoch_label(), fg=self.display.epoch_index_fg, bold=True),
            _style(self._current_phase().upper(), fg=self.display.phase_fg, bg=self.display.phase_bg, bold=True),
            _style(f"{batch_index:>3}/{total_batches:<3}", fg=self.display.batch_index_fg, bold=True),
            _style(bar, fg=self.display.progress_fg, bold=True),
            self._format_metric("loss", f"{metrics['loss']:.4f}", value_fg=self.display.metric_value_fg),
        ]
        if "reconstruction_loss" in metrics:
            parts.append(self._format_metric("recon", f"{metrics['reconstruction_loss']:.4f}", value_fg=self.display.recon_value_fg))
        if "sparsity_loss" in metrics:
            parts.append(self._format_metric("sparse", f"{metrics['sparsity_loss']:.4f}", value_fg=self.display.sparse_value_fg))
        if "kl_loss" in metrics:
            parts.append(self._format_metric("kl", f"{metrics['kl_loss']:.4f}", value_fg=self.display.kl_value_fg))
        self._write_live_line(self._join_segments(*parts))

    def _log_epoch_summary(
        self,
        epoch_metrics: dict[str, float | int],
        *,
        persist: bool,
    ) -> None:
        summary_parts = [
            _style(self._format_epoch_label(), fg=self.display.epoch_index_fg, bold=True),
            _style("EVAL", fg=self.display.phase_fg, bg=self.display.phase_bg, bold=True),
            self._format_metric("train", f"{float(epoch_metrics['train_loss']):.4f}", value_fg=self.display.metric_value_fg),
            self._format_metric("valid", f"{float(epoch_metrics['validation_loss']):.4f}", value_fg=self.display.metric_value_fg),
        ]

        if "train_reconstruction_loss" in epoch_metrics and "validation_reconstruction_loss" in epoch_metrics:
            summary_parts.append(
                self._format_metric(
                    "recon",
                    f"{float(epoch_metrics['train_reconstruction_loss']):.4f}/{float(epoch_metrics['validation_reconstruction_loss']):.4f}",
                    value_fg=self.display.recon_value_fg,
                )
            )
        if "train_sparsity_loss" in epoch_metrics and "validation_sparsity_loss" in epoch_metrics:
            summary_parts.append(
                self._format_metric(
                    "sparse",
                    f"{float(epoch_metrics['train_sparsity_loss']):.4f}/{float(epoch_metrics['validation_sparsity_loss']):.4f}",
                    value_fg=self.display.sparse_value_fg,
                )
            )
        if "train_kl_loss" in epoch_metrics and "validation_kl_loss" in epoch_metrics:
            summary_parts.append(
                self._format_metric(
                    "kl",
                    f"{float(epoch_metrics['train_kl_loss']):.4f}/{float(epoch_metrics['validation_kl_loss']):.4f}",
                    value_fg=self.display.kl_value_fg,
                )
            )
        if "train_free_bits_kl_loss" in epoch_metrics and "validation_free_bits_kl_loss" in epoch_metrics:
            summary_parts.append(
                self._format_metric(
                    "free-kl",
                    f"{float(epoch_metrics['train_free_bits_kl_loss']):.4f}/{float(epoch_metrics['validation_free_bits_kl_loss']):.4f}",
                    value_fg=self.display.free_kl_value_fg,
                )
            )

        line = self._join_segments(*summary_parts)
        if persist:
            self._print_log("EPOCH", line, fg=self.display.epoch_label_fg, bg=self.display.epoch_label_bg)
        else:
            self._write_live_line(line)

    def _log_best_epoch(self, epoch_metrics: dict[str, float | int]) -> None:
        parts = [
            _style(self._format_epoch_label(), fg=self.display.epoch_index_fg, bold=True),
            self._format_metric("valid", f"{float(epoch_metrics['validation_loss']):.4f}", value_fg=self.display.metric_value_fg),
        ]
        if "validation_reconstruction_loss" in epoch_metrics:
            parts.append(
                self._format_metric("recon", f"{float(epoch_metrics['validation_reconstruction_loss']):.4f}", value_fg=self.display.recon_value_fg)
            )
        if "validation_sparsity_loss" in epoch_metrics:
            parts.append(
                self._format_metric("sparse", f"{float(epoch_metrics['validation_sparsity_loss']):.4f}", value_fg=self.display.sparse_value_fg)
            )
        if "validation_kl_loss" in epoch_metrics:
            parts.append(self._format_metric("kl", f"{float(epoch_metrics['validation_kl_loss']):.4f}", value_fg=self.display.kl_value_fg))
        if "validation_free_bits_kl_loss" in epoch_metrics:
            parts.append(
                self._format_metric(
                    "free-kl",
                    f"{float(epoch_metrics['validation_free_bits_kl_loss']):.4f}",
                    value_fg=self.display.free_kl_value_fg,
                )
            )
        self._print_log("BEST", self._join_segments(*parts), fg=self.display.best_label_fg, bg=self.display.best_label_bg)

    def _log_run_end(
        self,
        test_metrics: dict[str, float],
        output_dir: Path,
        metrics_path: Path,
        stopped_early: bool,
        best_epoch: int | None,
    ) -> None:
        summary_parts = [self._format_metric("test", f"{test_metrics['loss']:.4f}", value_fg=self.display.metric_value_fg)]
        if "reconstruction_loss" in test_metrics:
            summary_parts.append(self._format_metric("recon", f"{test_metrics['reconstruction_loss']:.4f}", value_fg=self.display.recon_value_fg))
        if "sparsity_loss" in test_metrics:
            summary_parts.append(self._format_metric("sparse", f"{test_metrics['sparsity_loss']:.4f}", value_fg=self.display.sparse_value_fg))
        if "kl_loss" in test_metrics:
            summary_parts.append(self._format_metric("kl", f"{test_metrics['kl_loss']:.4f}", value_fg=self.display.kl_value_fg))
        if "free_bits_kl_loss" in test_metrics:
            summary_parts.append(self._format_metric("free-kl", f"{test_metrics['free_bits_kl_loss']:.4f}", value_fg=self.display.free_kl_value_fg))
        if stopped_early and best_epoch is not None:
            summary_parts.append(self._format_metric("best", str(best_epoch), value_fg=self.display.meta_value_fg))
            summary_parts.append(self._format_metric("stopped", str(self.current_epoch), value_fg=self.display.meta_value_fg))
        self._clear_live_line()
        self._print_log("FINAL", self._join_segments(*summary_parts), fg=self.display.final_label_fg, bg=self.display.final_label_bg)
        self._print_log("SAVE", f"best -> {output_dir / 'best'}", fg=self.display.save_label_fg, bg=self.display.save_label_bg)
        self._print_log("SAVE", f"final -> {output_dir / 'final'}", fg=self.display.save_label_fg, bg=self.display.save_label_bg)
        self._print_log("SAVE", f"metrics -> {metrics_path}", fg=self.display.save_label_fg, bg=self.display.save_label_bg)

    def _format_epoch_label(self) -> str:
        if self.max_epochs is None:
            return f"{self.current_epoch:03d}"
        return f"{self.current_epoch:03d}/{self.max_epochs:03d}"

    def _current_phase(self) -> str:
        return "train" if self.model.training else "eval"

    def _write_live_line(self, text: str) -> None:
        visible_length = _visible_len(text)
        padding = max(self._last_live_line_length - visible_length, 0)
        sys.stdout.write("\r" + text + (" " * padding))
        sys.stdout.flush()
        self._last_live_line_length = visible_length

    def _clear_live_line(self) -> None:
        if self._last_live_line_length == 0:
            return
        sys.stdout.write("\r" + (" " * self._last_live_line_length) + "\r")
        sys.stdout.flush()
        self._last_live_line_length = 0

    @staticmethod
    def _print_log(label: str, text: str, *, fg: str, bg: str) -> None:
        print(f"{_format_label(label, fg=fg, bg=bg)} {text}")

    def _format_metric(self, name: str, value: str, *, value_fg: str) -> str:
        return f"{_style(name, fg=self.display.metric_name_fg, dim=True)} {_style(value, fg=value_fg, bold=True)}"

    def _join_segments(self, *segments: str) -> str:
        separator = _style(self.display.separator, fg="magenta", dim=True)
        return separator.join(segment for segment in segments if segment)


class VAETrainer(AutoencoderTrainer):
    """Trainer with VAE-specific objective scheduling hooks."""

    def __init__(
        self,
        model,
        args: VAETrainingArguments,
        optimizer: torch.optim.Optimizer | None = None,
        display: TrainerDisplayConfig | None = None,
    ) -> None:
        super().__init__(model=model, args=args, optimizer=optimizer, display=display)

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
