"""Trainer abstractions for autoencoder-family models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..data.base import DatasetLoaders
from .display import TrainerDisplay, TrainerDisplayConfig


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
class AdversarialAutoencoderTrainingArguments(TrainingArguments):
    """Training settings specific to adversarial autoencoders."""

    discriminator_learning_rate: float | None = None
    generator_learning_rate: float | None = None
    discriminator_steps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.discriminator_learning_rate is not None and self.discriminator_learning_rate <= 0:
            raise ValueError("discriminator_learning_rate must be greater than 0 when provided.")
        if self.generator_learning_rate is not None and self.generator_learning_rate <= 0:
            raise ValueError("generator_learning_rate must be greater than 0 when provided.")
        if self.discriminator_steps <= 0:
            raise ValueError("discriminator_steps must be greater than 0.")


@dataclass
class FactorVariationalAutoencoderTrainingArguments(TrainingArguments):
    """Training settings specific to FactorVAE models."""

    discriminator_learning_rate: float | None = None
    discriminator_steps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.discriminator_learning_rate is not None and self.discriminator_learning_rate <= 0:
            raise ValueError("discriminator_learning_rate must be greater than 0 when provided.")
        if self.discriminator_steps <= 0:
            raise ValueError("discriminator_steps must be greater than 0.")


class AETrainer:
    """Shared trainer for deterministic autoencoder-style models."""

    def __init__(
        self,
        model,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer | None = None,
        display: TrainerDisplayConfig | TrainerDisplay | None = None,
    ) -> None:
        self.model = model
        self.args = args
        if isinstance(display, TrainerDisplay):
            self.display = display
        else:
            self.display = TrainerDisplay(display)
        self.current_epoch = 0
        self.global_step = 0
        self.max_epochs: int | None = None
        self.device = resolve_device(args.device)
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
        )

    def train_epoch(self, dataloader) -> dict[str, float]:
        self.model.train()
        return self.run_epoch(dataloader, training=True)

    def evaluate(self, dataloader) -> dict[str, float]:
        self.model.eval()
        if self.requires_grad_in_eval():
            return self.run_epoch(dataloader, training=False)
        with torch.no_grad():
            return self.run_epoch(dataloader, training=False)

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

        self.display.log_run_start(
            model_name=metadata.get("model", self.model.__class__.__name__) if metadata else self.model.__class__.__name__,
            dataset_name=metadata.get("dataset", "unknown") if metadata else "unknown",
            device=str(self.device),
            epoch_budget="early-stop" if max_epochs is None else str(max_epochs),
        )

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
                self.display.clear_live_line()
                self.display.log_best_epoch(
                    epoch_label=self.format_epoch_label(),
                    epoch_metrics=epoch_metrics,
                )
            else:
                if self.args.show_only_best_epochs:
                    self.display.log_epoch_summary(
                        epoch_label=self.format_epoch_label(),
                        epoch_metrics=epoch_metrics,
                        persist=False,
                    )
                else:
                    self.display.clear_live_line()
                    self.display.log_epoch_summary(
                        epoch_label=self.format_epoch_label(),
                        epoch_metrics=epoch_metrics,
                        persist=True,
                    )
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

        self.display.log_run_end(
            test_metrics=test_metrics,
            output_dir=output_dir,
            metrics_path=metrics_path,
            stopped_early=stopped_early,
            best_epoch=best_epoch,
            current_epoch=self.current_epoch,
        )
        return metrics

    def on_epoch_start(self, epoch: int) -> None:
        self.current_epoch = epoch

    def get_epoch_metrics(self) -> dict[str, float | int]:
        return self.model.get_epoch_metrics(
            global_step=self.global_step,
            current_epoch=self.current_epoch,
        )

    def requires_grad_in_eval(self) -> bool:
        return self.model.requires_grad_in_eval

    def run_epoch(self, dataloader, *, training: bool) -> dict[str, float]:
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
                self.global_step += 1

            batch_size = batch.shape[0]
            total_examples += batch_size
            batch_metrics = self.extract_batch_metrics(outputs, loss=loss)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            if training:
                running_metrics = {name: total / max(total_examples, 1) for name, total in totals.items()}
                self.display.log_batch_progress(
                    epoch_label=self.format_epoch_label(),
                    phase=self.current_phase(),
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metrics=running_metrics,
                )

        return {name: total / max(total_examples, 1) for name, total in totals.items()}

    def forward_batch(self, batch: torch.Tensor, *, training: bool):
        global_step = self.global_step + 1 if training else None
        return self.model(inputs=batch, global_step=global_step, current_epoch=self.current_epoch)

    def compute_batch_loss(self, outputs, *, training: bool) -> torch.Tensor:
        return outputs.loss

    def extract_batch_metrics(self, outputs, *, loss: torch.Tensor) -> dict[str, float]:
        metrics: dict[str, float] = {"loss": float(loss.detach().item())}
        for name, value in outputs.loss_dict.items():
            if name == "loss":
                continue
            metrics[name] = float(value.detach().item()) if hasattr(value, "detach") else float(value)
        return metrics

    def format_epoch_label(self) -> str:
        if self.max_epochs is None:
            return f"{self.current_epoch:03d}"
        return f"{self.current_epoch:03d}/{self.max_epochs:03d}"

    def current_phase(self) -> str:
        return "train" if self.model.training else "eval"


class VAETrainer(AETrainer):
    """Trainer for variational autoencoder families."""


class FactorVAETrainer(VAETrainer):
    """Trainer with alternating VAE and latent-discriminator updates for FactorVAE."""

    def __init__(
        self,
        model,
        args: FactorVariationalAutoencoderTrainingArguments,
        optimizer: torch.optim.Optimizer | None = None,
        display: TrainerDisplayConfig | TrainerDisplay | None = None,
    ) -> None:
        super().__init__(model=model, args=args, optimizer=optimizer, display=display)
        autoencoder_parameters = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        autoencoder_parameters.extend(self.model.mean_projection.parameters())
        autoencoder_parameters.extend(self.model.logvar_projection.parameters())
        self.optimizer = optimizer or torch.optim.Adam(autoencoder_parameters, lr=args.learning_rate)
        discriminator_learning_rate = args.discriminator_learning_rate or args.learning_rate
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=discriminator_learning_rate,
        )

    @property
    def factor_args(self) -> FactorVariationalAutoencoderTrainingArguments:
        return self.args

    def run_epoch(self, dataloader, *, training: bool) -> dict[str, float]:
        totals: dict[str, float] = {}
        total_examples = 0
        total_batches = len(dataloader)

        for batch_index, batch in enumerate(dataloader, start=1):
            batch = batch.to(self.device)
            batch_global_step = self.global_step + 1 if training else None

            if training:
                self.optimizer.zero_grad()
                outputs = self.model(inputs=batch, global_step=batch_global_step, current_epoch=self.current_epoch)
                outputs.loss.backward()
                self.optimizer.step()

                for _ in range(self.factor_args.discriminator_steps):
                    self.discriminator_optimizer.zero_grad()
                    with torch.no_grad():
                        posterior_mean, posterior_logvar = self.model.encode(batch)
                        latents = self.model.sample_latents(
                            posterior_mean=posterior_mean,
                            posterior_logvar=posterior_logvar,
                            sample_posterior=True,
                        )
                        permuted_latents = self.model.permute_dims(latents)
                    discriminator_loss = self.model.compute_discriminator_loss(latents, permuted_latents)
                    discriminator_loss.backward()
                    self.discriminator_optimizer.step()

                self.global_step += 1

            with torch.no_grad():
                outputs = self.model(inputs=batch, global_step=batch_global_step, current_epoch=self.current_epoch)

            batch_size = batch.shape[0]
            total_examples += batch_size
            batch_metrics = self.extract_batch_metrics(outputs, loss=outputs.loss)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            if training:
                running_metrics = {name: total / max(total_examples, 1) for name, total in totals.items()}
                self.display.log_batch_progress(
                    epoch_label=self.format_epoch_label(),
                    phase=self.current_phase(),
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metrics=running_metrics,
                )

        return {name: total / max(total_examples, 1) for name, total in totals.items()}


class VQTrainer(AETrainer):
    """Trainer with codebook usage evaluation for quantized autoencoders."""

    def run_epoch(self, dataloader, *, training: bool) -> dict[str, float]:
        totals: dict[str, float] = {}
        total_examples = 0
        total_batches = len(dataloader)
        code_counts = self.initialize_code_counts()

        for batch_index, batch in enumerate(dataloader, start=1):
            batch = batch.to(self.device)
            if training:
                self.optimizer.zero_grad()
            outputs = self.forward_batch(
                batch,
                training=training,
                is_last_train_step=training and batch_index == total_batches,
            )
            loss = self.compute_batch_loss(outputs, training=training)
            if training:
                loss.backward()
                self.optimizer.step()
                self.global_step += 1

            batch_size = batch.shape[0]
            total_examples += batch_size
            code_counts = self.accumulate_code_counts(code_counts, outputs.codebook_indices)
            batch_metrics = self.extract_batch_metrics(outputs, loss=loss)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            if training:
                running_metrics = {name: total / max(total_examples, 1) for name, total in totals.items()}
                self.display.log_batch_progress(
                    epoch_label=self.format_epoch_label(),
                    phase=self.current_phase(),
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metrics=running_metrics,
                )

        metrics = {name: total / max(total_examples, 1) for name, total in totals.items()}
        metrics.update(self.compute_codebook_metrics(code_counts))
        if training:
            metrics["dead_code_reset_count"] = float(self.model.consume_dead_code_reset_count())
        return metrics

    def forward_batch(
        self,
        batch: torch.Tensor,
        *,
        training: bool,
        is_last_train_step: bool = False,
    ):
        global_step = self.global_step + 1 if training else None
        return self.model(
            inputs=batch,
            global_step=global_step,
            current_epoch=self.current_epoch,
            is_last_train_step=is_last_train_step if training else None,
        )

    def initialize_code_counts(self) -> torch.Tensor:
        codebook_size = int(self.model.config.codebook_size)
        return torch.zeros(codebook_size, dtype=torch.long, device=self.device)

    def accumulate_code_counts(self, code_counts: torch.Tensor, codebook_indices: torch.Tensor) -> torch.Tensor:
        if codebook_indices.ndim == 1:
            code_counts += torch.bincount(codebook_indices.reshape(-1), minlength=code_counts.shape[-1])
            return code_counts

        if code_counts.ndim == 1:
            code_counts = torch.zeros(
                codebook_indices.shape[1],
                code_counts.shape[0],
                dtype=code_counts.dtype,
                device=code_counts.device,
            )
        for codebook_index in range(codebook_indices.shape[1]):
            code_counts[codebook_index] += torch.bincount(
                codebook_indices[:, codebook_index].reshape(-1),
                minlength=code_counts.shape[-1],
            )
        return code_counts

    def compute_codebook_metrics(self, code_counts: torch.Tensor) -> dict[str, float]:
        if code_counts.ndim == 2:
            active_counts = (code_counts > 0).sum(dim=1)
            active_codes = int(active_counts.sum().item())
            codebook_size = int(code_counts.numel())
            usage_ratio = active_codes / codebook_size if codebook_size > 0 else 0.0
            dead_code_ratio = 1.0 - usage_ratio

            perplexities: list[float] = []
            for counts_for_book in code_counts:
                total_for_book = float(counts_for_book.sum().item())
                if total_for_book == 0:
                    perplexities.append(0.0)
                    continue
                probabilities = counts_for_book.to(dtype=torch.float32) / total_for_book
                used_probabilities = probabilities[probabilities > 0]
                entropy = -(used_probabilities * used_probabilities.log()).sum()
                perplexities.append(float(entropy.exp().item()))

            return {
                "active_codes": float(active_codes),
                "codebook_size": float(codebook_size),
                "codebook_usage_ratio": usage_ratio,
                "dead_code_ratio": dead_code_ratio,
                "codebook_perplexity": sum(perplexities) / max(len(perplexities), 1),
            }

        counts = code_counts.detach().to(dtype=torch.float32)
        total_codes = float(counts.sum().item())
        active_codes = int((counts > 0).sum().item())
        codebook_size = int(counts.numel())
        usage_ratio = active_codes / codebook_size if codebook_size > 0 else 0.0
        dead_code_ratio = 1.0 - usage_ratio

        if total_codes == 0:
            perplexity = 0.0
        else:
            probabilities = counts / total_codes
            used_probabilities = probabilities[probabilities > 0]
            entropy = -(used_probabilities * used_probabilities.log()).sum()
            perplexity = float(entropy.exp().item())

        return {
            "active_codes": float(active_codes),
            "codebook_size": float(codebook_size),
            "codebook_usage_ratio": usage_ratio,
            "dead_code_ratio": dead_code_ratio,
            "codebook_perplexity": perplexity,
        }


class AdversarialAutoencoderTrainer(AETrainer):
    """Trainer with alternating reconstruction, discriminator, and encoder-adversarial updates."""

    def __init__(
        self,
        model,
        args: AdversarialAutoencoderTrainingArguments,
        optimizer: torch.optim.Optimizer | None = None,
        display: TrainerDisplayConfig | TrainerDisplay | None = None,
    ) -> None:
        super().__init__(model=model, args=args, optimizer=optimizer, display=display)
        autoencoder_parameters = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        self.optimizer = optimizer or torch.optim.Adam(autoencoder_parameters, lr=args.learning_rate)
        generator_learning_rate = args.generator_learning_rate or args.learning_rate
        discriminator_learning_rate = args.discriminator_learning_rate or args.learning_rate
        self.generator_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=generator_learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=discriminator_learning_rate,
        )

    @property
    def adversarial_args(self) -> AdversarialAutoencoderTrainingArguments:
        return self.args

    def run_epoch(self, dataloader, *, training: bool) -> dict[str, float]:
        totals: dict[str, float] = {}
        total_examples = 0
        total_batches = len(dataloader)

        for batch_index, batch in enumerate(dataloader, start=1):
            batch = batch.to(self.device)
            batch_global_step = self.global_step + 1 if training else None

            if training:
                self.optimizer.zero_grad()
                latents = self.model.encode(batch)
                reconstruction = self.model.decode(latents)
                reconstruction_loss = self.model.compute_loss(reconstruction, batch)
                reconstruction_loss.backward()
                self.optimizer.step()

                for _ in range(self.adversarial_args.discriminator_steps):
                    self.discriminator_optimizer.zero_grad()
                    with torch.no_grad():
                        detached_latents = self.model.encode(batch)
                    prior_samples = self.model.sample_prior(
                        batch.shape[0],
                        device=batch.device,
                        dtype=batch.dtype,
                    )
                    discriminator_loss = self.model.compute_discriminator_loss(detached_latents, prior_samples)
                    discriminator_loss.backward()
                    self.discriminator_optimizer.step()

                self.generator_optimizer.zero_grad()
                adversarial_latents = self.model.encode(batch)
                adversarial_loss = self.model.compute_adversarial_loss(adversarial_latents)
                (self.model.config.adversarial_weight * adversarial_loss).backward()
                self.generator_optimizer.step()
                self.global_step += 1

            with torch.no_grad():
                outputs = self.model(inputs=batch, global_step=batch_global_step, current_epoch=self.current_epoch)

            batch_size = batch.shape[0]
            total_examples += batch_size
            batch_metrics = self.extract_batch_metrics(outputs, loss=outputs.loss)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            if training:
                running_metrics = {name: total / max(total_examples, 1) for name, total in totals.items()}
                self.display.log_batch_progress(
                    epoch_label=self.format_epoch_label(),
                    phase=self.current_phase(),
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metrics=running_metrics,
                )

        return {name: total / max(total_examples, 1) for name, total in totals.items()}
