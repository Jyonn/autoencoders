"""Terminal display helpers for trainer logs."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


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


def style(
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


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def format_label(name: str, *, fg: str, bg: str) -> str:
    return style(f" {name} ", fg=fg, bg=bg, bold=True)


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
    contractive_value_fg: str = "blue"
    free_kl_value_fg: str = "blue"
    mmd_value_fg: str = "yellow"
    adversarial_value_fg: str = "magenta"
    discriminator_value_fg: str = "red"
    commitment_value_fg: str = "cyan"
    codebook_value_fg: str = "red"
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
            self.contractive_value_fg,
            self.free_kl_value_fg,
            self.mmd_value_fg,
            self.adversarial_value_fg,
            self.discriminator_value_fg,
            self.commitment_value_fg,
            self.codebook_value_fg,
            self.meta_value_fg,
        ):
            if color_name not in FG and color_name not in BG:
                raise ValueError(f"Unsupported display color: {color_name}")


class TrainerDisplay:
    """Stateful terminal logger for trainer progress and summaries."""

    def __init__(self, config: TrainerDisplayConfig | None = None) -> None:
        self.config = config or TrainerDisplayConfig()
        self._last_live_line_length = 0

    def log_run_start(self, *, model_name: str, dataset_name: str, device: str, epoch_budget: str) -> None:
        self._print_log(
            "RUN",
            self._join_segments(
                self.format_metric("model", model_name, value_fg=self.config.metric_value_fg),
                self.format_metric("dataset", dataset_name, value_fg=self.config.metric_value_fg),
                self.format_metric("device", device, value_fg=self.config.meta_value_fg),
                self.format_metric("epochs", epoch_budget, value_fg=self.config.progress_fg),
            ),
            fg=self.config.run_label_fg,
            bg=self.config.run_label_bg,
        )

    def log_batch_progress(
        self,
        *,
        epoch_label: str,
        phase: str,
        batch_index: int,
        total_batches: int,
        metrics: dict[str, float],
    ) -> None:
        progress_width = self.config.progress_width
        filled = int(progress_width * batch_index / max(total_batches, 1))
        bar = self.config.progress_fill * filled + self.config.progress_empty * (progress_width - filled)
        parts = [
            style(epoch_label, fg=self.config.epoch_index_fg, bold=True),
            style(phase.upper(), fg=self.config.phase_fg, bg=self.config.phase_bg, bold=True),
            style(f"{batch_index:>3}/{total_batches:<3}", fg=self.config.batch_index_fg, bold=True),
            style(bar, fg=self.config.progress_fg, bold=True),
            self.format_metric("loss", f"{metrics['loss']:.4f}", value_fg=self.config.metric_value_fg),
        ]
        if "reconstruction_loss" in metrics:
            parts.append(self.format_metric("recon", f"{metrics['reconstruction_loss']:.4f}", value_fg=self.config.recon_value_fg))
        if "sparsity_loss" in metrics:
            parts.append(self.format_metric("sparse", f"{metrics['sparsity_loss']:.4f}", value_fg=self.config.sparse_value_fg))
        if "topk_sparsity" in metrics:
            parts.append(self.format_metric("topk", f"{metrics['topk_sparsity']:.4f}", value_fg=self.config.sparse_value_fg))
        if "kl_sparsity_loss" in metrics:
            parts.append(self.format_metric("kl-sparse", f"{metrics['kl_sparsity_loss']:.4f}", value_fg=self.config.sparse_value_fg))
        if "contractive_loss" in metrics:
            parts.append(self.format_metric("contract", f"{metrics['contractive_loss']:.4f}", value_fg=self.config.contractive_value_fg))
        if "mmd_loss" in metrics:
            parts.append(self.format_metric("mmd", f"{metrics['mmd_loss']:.4f}", value_fg=self.config.mmd_value_fg))
        if "adversarial_loss" in metrics:
            parts.append(self.format_metric("adv", f"{metrics['adversarial_loss']:.4f}", value_fg=self.config.adversarial_value_fg))
        if "discriminator_loss" in metrics:
            parts.append(self.format_metric("disc", f"{metrics['discriminator_loss']:.4f}", value_fg=self.config.discriminator_value_fg))
        if "commitment_loss" in metrics:
            parts.append(self.format_metric("commit", f"{metrics['commitment_loss']:.4f}", value_fg=self.config.commitment_value_fg))
        if "codebook_loss" in metrics:
            parts.append(self.format_metric("book", f"{metrics['codebook_loss']:.4f}", value_fg=self.config.codebook_value_fg))
        if "kl_loss" in metrics:
            parts.append(self.format_metric("kl", f"{metrics['kl_loss']:.4f}", value_fg=self.config.kl_value_fg))
        self.write_live_line(self._join_segments(*parts))

    def log_epoch_summary(self, *, epoch_label: str, epoch_metrics: dict[str, float | int], persist: bool) -> None:
        summary_parts = [
            style(epoch_label, fg=self.config.epoch_index_fg, bold=True),
            style("EVAL", fg=self.config.phase_fg, bg=self.config.phase_bg, bold=True),
            self.format_metric("train", f"{float(epoch_metrics['train_loss']):.4f}", value_fg=self.config.metric_value_fg),
            self.format_metric("valid", f"{float(epoch_metrics['validation_loss']):.4f}", value_fg=self.config.metric_value_fg),
        ]
        summary_parts.extend(self._summary_metric_segments(epoch_metrics, train_validation=True))
        line = self._join_segments(*summary_parts)
        if persist:
            self._print_log("EPOCH", line, fg=self.config.epoch_label_fg, bg=self.config.epoch_label_bg)
        else:
            self.write_live_line(line)

    def log_best_epoch(self, *, epoch_label: str, epoch_metrics: dict[str, float | int]) -> None:
        parts = [
            style(epoch_label, fg=self.config.epoch_index_fg, bold=True),
            self.format_metric("valid", f"{float(epoch_metrics['validation_loss']):.4f}", value_fg=self.config.metric_value_fg),
        ]
        parts.extend(self._summary_metric_segments(epoch_metrics, train_validation=False))
        self._print_log("BEST", self._join_segments(*parts), fg=self.config.best_label_fg, bg=self.config.best_label_bg)

    def log_run_end(
        self,
        *,
        test_metrics: dict[str, float],
        output_dir: Path,
        metrics_path: Path,
        stopped_early: bool,
        best_epoch: int | None,
        current_epoch: int,
    ) -> None:
        summary_parts = [self.format_metric("test", f"{test_metrics['loss']:.4f}", value_fg=self.config.metric_value_fg)]
        summary_parts.extend(self._test_metric_segments(test_metrics))
        if stopped_early and best_epoch is not None:
            summary_parts.append(self.format_metric("best", str(best_epoch), value_fg=self.config.meta_value_fg))
            summary_parts.append(self.format_metric("stopped", str(current_epoch), value_fg=self.config.meta_value_fg))
        self.clear_live_line()
        self._print_log("FINAL", self._join_segments(*summary_parts), fg=self.config.final_label_fg, bg=self.config.final_label_bg)
        self._print_log("SAVE", f"best -> {output_dir / 'best'}", fg=self.config.save_label_fg, bg=self.config.save_label_bg)
        self._print_log("SAVE", f"final -> {output_dir / 'final'}", fg=self.config.save_label_fg, bg=self.config.save_label_bg)
        self._print_log("SAVE", f"metrics -> {metrics_path}", fg=self.config.save_label_fg, bg=self.config.save_label_bg)

    def format_metric(self, name: str, value: str, *, value_fg: str) -> str:
        return f"{style(name, fg=self.config.metric_name_fg, dim=True)} {style(value, fg=value_fg, bold=True)}"

    def write_live_line(self, text: str) -> None:
        visible_length = visible_len(text)
        padding = max(self._last_live_line_length - visible_length, 0)
        sys.stdout.write("\r" + text + (" " * padding))
        sys.stdout.flush()
        self._last_live_line_length = visible_length

    def clear_live_line(self) -> None:
        if self._last_live_line_length == 0:
            return
        sys.stdout.write("\r" + (" " * self._last_live_line_length) + "\r")
        sys.stdout.flush()
        self._last_live_line_length = 0

    def _join_segments(self, *segments: str) -> str:
        separator = style(self.config.separator, fg="magenta", dim=True)
        return separator.join(segment for segment in segments if segment)

    @staticmethod
    def _print_log(label: str, text: str, *, fg: str, bg: str) -> None:
        print(f"{format_label(label, fg=fg, bg=bg)} {text}")

    def _summary_metric_segments(
        self,
        epoch_metrics: dict[str, float | int],
        *,
        train_validation: bool,
    ) -> list[str]:
        if train_validation:
            def value(name: str) -> str:
                return f"{float(epoch_metrics[f'train_{name}']):.4f}/{float(epoch_metrics[f'validation_{name}']):.4f}"
            prefix = "validation_"
        else:
            def value(name: str) -> str:
                return f"{float(epoch_metrics[f'validation_{name}']):.4f}"
            prefix = "validation_"

        parts: list[str] = []
        pairs = [
            ("reconstruction_loss", "recon", self.config.recon_value_fg),
            ("sparsity_loss", "sparse", self.config.sparse_value_fg),
            ("topk_sparsity", "topk", self.config.sparse_value_fg),
            ("kl_sparsity_loss", "kl-sparse", self.config.sparse_value_fg),
            ("contractive_loss", "contract", self.config.contractive_value_fg),
            ("mmd_loss", "mmd", self.config.mmd_value_fg),
            ("adversarial_loss", "adv", self.config.adversarial_value_fg),
            ("discriminator_loss", "disc", self.config.discriminator_value_fg),
            ("commitment_loss", "commit", self.config.commitment_value_fg),
            ("codebook_loss", "book", self.config.codebook_value_fg),
            ("kl_loss", "kl", self.config.kl_value_fg),
            ("free_bits_kl_loss", "free-kl", self.config.free_kl_value_fg),
        ]
        for metric_name, label, color in pairs:
            if f"{prefix}{metric_name}" in epoch_metrics:
                parts.append(self.format_metric(label, value(metric_name), value_fg=color))

        if "validation_active_codes" in epoch_metrics and "validation_codebook_size" in epoch_metrics:
            parts.append(
                self.format_metric(
                    "codes",
                    f"{int(epoch_metrics['validation_active_codes'])}/{int(epoch_metrics['validation_codebook_size'])}",
                    value_fg=self.config.meta_value_fg,
                )
            )
        if "validation_codebook_usage_ratio" in epoch_metrics:
            parts.append(
                self.format_metric(
                    "usage",
                    f"{float(epoch_metrics['validation_codebook_usage_ratio']):.3f}",
                    value_fg=self.config.meta_value_fg,
                )
            )
        if "validation_codebook_perplexity" in epoch_metrics:
            parts.append(
                self.format_metric(
                    "ppl",
                    f"{float(epoch_metrics['validation_codebook_perplexity']):.2f}",
                    value_fg=self.config.meta_value_fg,
                )
            )
        if "validation_dead_code_ratio" in epoch_metrics:
            parts.append(
                self.format_metric(
                    "dead",
                    f"{float(epoch_metrics['validation_dead_code_ratio']):.3f}",
                    value_fg=self.config.meta_value_fg,
                )
            )
        return parts

    def _test_metric_segments(self, test_metrics: dict[str, float]) -> list[str]:
        parts: list[str] = []
        pairs = [
            ("reconstruction_loss", "recon", self.config.recon_value_fg),
            ("sparsity_loss", "sparse", self.config.sparse_value_fg),
            ("topk_sparsity", "topk", self.config.sparse_value_fg),
            ("kl_sparsity_loss", "kl-sparse", self.config.sparse_value_fg),
            ("contractive_loss", "contract", self.config.contractive_value_fg),
            ("mmd_loss", "mmd", self.config.mmd_value_fg),
            ("adversarial_loss", "adv", self.config.adversarial_value_fg),
            ("discriminator_loss", "disc", self.config.discriminator_value_fg),
            ("commitment_loss", "commit", self.config.commitment_value_fg),
            ("codebook_loss", "book", self.config.codebook_value_fg),
            ("kl_loss", "kl", self.config.kl_value_fg),
            ("free_bits_kl_loss", "free-kl", self.config.free_kl_value_fg),
        ]
        for metric_name, label, color in pairs:
            if metric_name in test_metrics:
                parts.append(self.format_metric(label, f"{test_metrics[metric_name]:.4f}", value_fg=color))
        if "active_codes" in test_metrics and "codebook_size" in test_metrics:
            parts.append(
                self.format_metric(
                    "codes",
                    f"{int(test_metrics['active_codes'])}/{int(test_metrics['codebook_size'])}",
                    value_fg=self.config.meta_value_fg,
                )
            )
        if "codebook_usage_ratio" in test_metrics:
            parts.append(self.format_metric("usage", f"{test_metrics['codebook_usage_ratio']:.3f}", value_fg=self.config.meta_value_fg))
        if "codebook_perplexity" in test_metrics:
            parts.append(self.format_metric("ppl", f"{test_metrics['codebook_perplexity']:.2f}", value_fg=self.config.meta_value_fg))
        if "dead_code_ratio" in test_metrics:
            parts.append(self.format_metric("dead", f"{test_metrics['dead_code_ratio']:.3f}", value_fg=self.config.meta_value_fg))
        return parts
