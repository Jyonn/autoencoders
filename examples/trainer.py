"""Train autoencoder-family models from a YAML configuration."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders.data.base import DataSpec
from autoencoders.data.loading import get_dataset_class, load_dataset
from autoencoders.function import set_seed
from autoencoders.models.loading import get_model_class
from autoencoders.training.display import style
from autoencoders.training.trainer import (
    AETrainer,
    AdversarialAutoencoderTrainer,
    AdversarialAutoencoderTrainingConfig,
    FactorVAETrainer,
    FactorVariationalAutoencoderTrainingConfig,
    TrainingConfig,
    VAETrainer,
    VQTrainer,
)
from examples.utils.config_init import ConfigInit


_LEVEL_COLORS = ("cyan", "magenta", "blue", "green")
_VAE_MODEL_NAMES = {
    "vae",
    "betavae",
    "dvae",
    "hvae",
    "vamppriorvae",
    "infovae",
    "mmdvae",
    "dipvae",
    "betatcvae",
}
_VQ_MODEL_NAMES = {
    "vqvae",
    "gumbelvq",
    "fsq",
    "rfsq",
    "pqvae",
    "rqvae",
    "vqvae2",
}


def _mapping_or_empty(value) -> dict[str, Any]:
    if not value:
        return {}
    return value()


def _build_effective_config(configurations) -> dict[str, Any]:
    return {
        "dataset": {
            "name": configurations.dataset.name,
            "config": _mapping_or_empty(configurations.dataset.config),
        },
        "model": {
            "name": configurations.model.name,
            "config": _mapping_or_empty(configurations.model.config),
        },
        "encoder": {
            "name": configurations.encoder.name or None,
            "config": _mapping_or_empty(configurations.encoder.config),
        },
        "decoder": (
            {
                "name": configurations.decoder.name or None,
                "config": _mapping_or_empty(configurations.decoder.config),
            }
            if configurations.decoder
            else None
        ),
        "trainer": _mapping_or_empty(configurations.trainer),
    }


def _render_scalar(value: Any) -> str:
    if value is None:
        return style("null", fg="yellow", dim=True)
    if isinstance(value, bool):
        return style("true" if value else "false", fg="yellow", bold=True)
    if isinstance(value, (int, float)):
        return style(str(value), fg="green", bold=True)
    return style(str(value), fg="white")


def _render_config_lines(value: Any, *, depth: int = 0) -> list[str]:
    indent = "  " * depth
    key_color = _LEVEL_COLORS[depth % len(_LEVEL_COLORS)]
    lines: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            rendered_key = style(str(key), fg=key_color, bold=True)
            if isinstance(child, dict):
                if not child:
                    lines.append(f"{indent}{rendered_key}: {style('{}', fg='white', dim=True)}")
                else:
                    lines.append(f"{indent}{rendered_key}:")
                    lines.extend(_render_config_lines(child, depth=depth + 1))
            elif isinstance(child, list):
                if not child:
                    lines.append(f"{indent}{rendered_key}: {style('[]', fg='white', dim=True)}")
                else:
                    lines.append(f"{indent}{rendered_key}:")
                    lines.extend(_render_config_lines(child, depth=depth + 1))
            else:
                lines.append(f"{indent}{rendered_key}: {_render_scalar(child)}")
        return lines

    if isinstance(value, list):
        dash = style("-", fg=key_color, bold=True)
        for child in value:
            if isinstance(child, (dict, list)):
                lines.append(f"{indent}{dash}")
                lines.extend(_render_config_lines(child, depth=depth + 1))
            else:
                lines.append(f"{indent}{dash} {_render_scalar(child)}")
        return lines

    return [f"{indent}{_render_scalar(value)}"]


def _print_effective_config(configurations) -> None:
    print()
    print(style(" Effective Config ", fg="white", bg="blue", bold=True))
    for line in _render_config_lines(_build_effective_config(configurations)):
        print(line)
    print(style(" End Config ", fg="black", bg="yellow", bold=True))
    print()


def _format_spec(spec: DataSpec) -> str:
    return style(str(spec), fg="green")


def _print_pipeline_trace(model) -> None:
    if not hasattr(model, "get_pipeline_trace"):
        return

    print(style(" Shape Trace ", fg="white", bg="magenta", bold=True))
    pipeline = model.get_pipeline_trace()
    if not pipeline:
        print(style("  <empty>", fg="yellow", dim=True))
        print(style(" End Trace ", fg="black", bg="yellow", bold=True))
        print()
        return

    first_step = pipeline[0]
    print(
        f"{style(first_step.name, fg='cyan', bold=True)} "
        f"{style(':', fg='magenta', dim=True)} "
        f"{_format_spec(first_step.output_spec)}"
    )

    for step in pipeline[1:]:
        header = (
            f"{style(step.name, fg='cyan', bold=True)} "
            f"{style('->', fg='magenta', dim=True)} "
            f"{_format_spec(step.output_spec)}"
        )
        print(header)
        for child in step.children or []:
            child_line = (
                f"  {style('↳', fg='yellow', bold=True)} "
                f"{style(child.name, fg='blue')} "
                f"{style('->', fg='magenta', dim=True)} "
                f"{_format_spec(child.output_spec)}"
            )
            print(child_line)
    print(style(" End Trace ", fg="black", bg="yellow", bold=True))
    print()


def select_trainer_components(model_name: str):
    if model_name == "factorvae":
        return FactorVAETrainer, FactorVariationalAutoencoderTrainingConfig
    if model_name == "aae":
        return AdversarialAutoencoderTrainer, AdversarialAutoencoderTrainingConfig
    if model_name in _VQ_MODEL_NAMES:
        return VQTrainer, TrainingConfig
    if model_name in _VAE_MODEL_NAMES:
        return VAETrainer, TrainingConfig
    return AETrainer, TrainingConfig


def build_model(configurations, sample_spec: DataSpec):
    model_class = get_model_class(configurations.model.name)
    model_config = model_class.config_class(**configurations.model.config())
    return model_class(
        config=model_config,
        sample_spec=sample_spec,
        encoder=configurations.encoder.name or None,
        encoder_config=configurations.encoder.config() if configurations.encoder.config else None,
        decoder=configurations.decoder.name or None,
        decoder_config=configurations.decoder.config() if configurations.decoder.config else None,
    )


def build_trainer(configurations, model):
    trainer_class, trainer_config_class = select_trainer_components(configurations.model.name)
    trainer_args = trainer_config_class(**configurations.trainer())
    return trainer_class(model=model, args=trainer_args)


def main() -> None:
    configurations = ConfigInit(
        required_args=["config"],
        default_args=[],
        makedirs=[],
    ).parse().config

    _print_effective_config(configurations)

    set_seed(configurations.trainer.seed)
    dataset_class = get_dataset_class(configurations.dataset.name)
    dataset = load_dataset(
        configurations.dataset.name,
        config=dataset_class.config_class(**_mapping_or_empty(configurations.dataset.config)),
    )

    dataloaders = dataset.get_dataloaders(
        batch_size=configurations.trainer.batch_size,
        validation_ratio=configurations.trainer.validation_ratio,
        test_ratio=configurations.trainer.test_ratio,
        seed=configurations.trainer.seed,
    )

    sample_spec = dataset.get_sample_spec()
    model = build_model(configurations, sample_spec)
    _print_pipeline_trace(model)

    trainer = build_trainer(configurations, model)
    trainer.fit(dataloaders, metadata={"dataset": configurations.dataset.name, "model": configurations.model.name})


if __name__ == "__main__":
    main()
