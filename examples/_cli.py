"""Helpers for dotted training CLI configuration."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import sys
from inspect import signature
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders.data.loading import get_dataset_class
from autoencoders.models.loading import get_model_class
from autoencoders.modules.loading import get_module_class


def _normalize_name(name: str) -> str:
    return name.replace("-", "_")


def _parse_cli_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def _collect_declared_config_fields(config_class) -> list[str]:
    fields: list[str] = []
    for current_class in reversed(config_class.mro()):
        init = current_class.__dict__.get("__init__")
        if init is None:
            continue
        for parameter_name, parameter in signature(init).parameters.items():
            if parameter_name in {"self", "kwargs"}:
                continue
            if parameter.kind is parameter.VAR_KEYWORD:
                continue
            if parameter_name not in fields:
                fields.append(parameter_name)
    return fields


def _validate_config_dict(values: dict[str, Any], config_class, *, scope: str) -> None:
    declared_fields = set(_collect_declared_config_fields(config_class))
    for key in values:
        if key in declared_fields:
            continue
        suggestion = difflib.get_close_matches(key, declared_fields, n=1)
        suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown {scope} option {key!r}.{suffix}")


def _default_config_dict(config_class) -> dict[str, Any]:
    config = config_class()
    return {
        field_name: getattr(config, field_name)
        for field_name in _collect_declared_config_fields(config_class)
        if hasattr(config, field_name)
    }


def _parse_dotted_overrides(tokens: list[str]) -> dict[str, dict[str, Any]]:
    overrides = {"dataset": {}, "model": {}, "encoder": {}, "decoder": {}, "trainer": {}}
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument {token!r}.")
        if "=" in token:
            flag, raw_value = token.split("=", 1)
            value = _parse_cli_value(raw_value)
            consumed = 1
        else:
            if index + 1 >= len(tokens):
                raise ValueError(f"Expected a value after {token!r}.")
            flag = token
            value = _parse_cli_value(tokens[index + 1])
            consumed = 2
        dotted_name = flag[2:]
        if "." not in dotted_name:
            raise ValueError(f"Unknown argument {flag!r}.")
        prefix, field_name = dotted_name.split(".", 1)
        prefix = _normalize_name(prefix)
        field_name = _normalize_name(field_name)
        if prefix not in overrides:
            raise ValueError(f"Unsupported dotted argument prefix {prefix!r}.")
        overrides[prefix][field_name] = value
        index += consumed

    return overrides


class ResolvedConfigArguments:
    def __init__(
        self,
        *,
        dataset_config: dict[str, Any],
        model_config: dict[str, Any],
        encoder_config: dict[str, Any] | None,
        decoder_config: dict[str, Any] | None,
        trainer_config: dict[str, Any],
    ) -> None:
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.trainer_config = trainer_config


TRAINING_ARGUMENT_FIELDS = {
    "output_dir",
    "validation_ratio",
    "test_ratio",
    "seed",
    "epochs",
    "patience",
    "batch_size",
    "learning_rate",
    "device",
    "show_only_best_epochs",
    "advice",
}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level YAML mapping in {path}.")
    return payload


def _pop_named_section(payload: dict[str, Any], name: str) -> tuple[str | None, dict[str, Any]]:
    section = payload.get(name)
    if section is None:
        return None, {}
    if isinstance(section, str):
        return section, {}
    if not isinstance(section, dict):
        raise ValueError(f"Expected `{name}` to be either a string or a mapping.")
    values = dict(section)
    raw_name = values.pop("name", None)
    if raw_name is not None and not isinstance(raw_name, str):
        raise ValueError(f"Expected `{name}.name` to be a string when provided.")
    return raw_name, values


def parse_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_dataset_config: dict[str, Any] | None = None,
    default_trainer_config: dict[str, Any] | None = None,
    default_model_config: dict[str, Any],
    default_encoder: str | None,
    default_encoder_config: dict[str, Any] | None,
) -> argparse.Namespace:
    args, unknown = parser.parse_known_args(sys.argv[1:])
    overrides = _parse_dotted_overrides(unknown)

    dataset_name = getattr(args, "dataset", None)
    if dataset_name is None:
        dataset_config = {**(default_dataset_config or {}), **overrides["dataset"]}
    else:
        dataset_class = get_dataset_class(dataset_name)
        declared_dataset_fields = set(_collect_declared_config_fields(dataset_class.config_class))
        seeded_defaults = {
            key: value
            for key, value in (default_dataset_config or {}).items()
            if key in declared_dataset_fields
        }
        dataset_config = {
            **_default_config_dict(dataset_class.config_class),
            **seeded_defaults,
            **overrides["dataset"],
        }
        _validate_config_dict(dataset_config, dataset_class.config_class, scope="dataset")

    trainer_config = {**(default_trainer_config or {}), **overrides["trainer"]}
    if default_trainer_config is not None:
        allowed_trainer_fields = set(default_trainer_config)
        for key in trainer_config:
            if key in allowed_trainer_fields:
                continue
            suggestion = difflib.get_close_matches(key, allowed_trainer_fields, n=1)
            suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
            raise ValueError(f"Unknown trainer option {key!r}.{suffix}")

    args.encoder = args.encoder or default_encoder
    model_class = get_model_class(args.model)
    model_config = {**default_model_config, **overrides["model"]}
    _validate_config_dict(model_config, model_class.config_class, scope="model")

    encoder_config: dict[str, Any] | None = None
    if args.encoder is not None:
        encoder_class = get_module_class(args.encoder)
        encoder_config = {
            **(default_encoder_config or {}),
            **overrides["encoder"],
        }
        _validate_config_dict(encoder_config, encoder_class.config_class, scope="encoder")
    elif overrides["encoder"]:
        raise ValueError("Received encoder.* options, but no encoder backbone was selected.")

    decoder_config: dict[str, Any] | None = None
    if args.decoder is not None:
        decoder_class = get_module_class(args.decoder)
        decoder_config = dict(overrides["decoder"])
        if not decoder_config and args.encoder is not None and args.decoder == args.encoder and encoder_config is not None:
            decoder_config = dict(encoder_config)
        _validate_config_dict(decoder_config, decoder_class.config_class, scope="decoder")
    elif overrides["decoder"]:
        raise ValueError("Received decoder.* options, but no decoder backbone was selected.")

    args.resolved_configs = ResolvedConfigArguments(
        dataset_config=dataset_config,
        model_config=model_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        trainer_config=trainer_config,
    )
    return args


def parse_yaml_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_dataset_config: dict[str, Any] | None = None,
    default_trainer_config: dict[str, Any] | None = None,
    default_model_config: dict[str, Any],
    default_encoder: str | None,
    default_encoder_config: dict[str, Any] | None,
    validate_sections: bool = True,
) -> argparse.Namespace:
    args, unknown = parser.parse_known_args(sys.argv[1:])
    config_path = getattr(args, "config", None)
    if config_path is None:
        raise ValueError("parse_yaml_config_arguments() requires args.config to be set.")
    if unknown:
        unexpected = " ".join(unknown)
        raise ValueError(f"Unexpected extra CLI arguments for YAML mode: {unexpected}")

    payload = _load_yaml_mapping(Path(config_path))
    overrides = _parse_dotted_overrides(unknown)

    dataset_name, dataset_values = _pop_named_section(payload, "dataset")
    model_name, model_values = _pop_named_section(payload, "model")
    encoder_name, encoder_values = _pop_named_section(payload, "encoder")
    decoder_name, decoder_values = _pop_named_section(payload, "decoder")
    trainer_values = payload.get("trainer") or {}
    if not isinstance(trainer_values, dict):
        raise ValueError("Expected `trainer` to be a mapping when provided.")
    trainer_values = dict(trainer_values)

    args.dataset = dataset_name or getattr(args, "dataset", None)
    args.model = model_name or getattr(args, "model", None)
    args.encoder = encoder_name if encoder_name is not None else default_encoder
    args.decoder = decoder_name

    if args.dataset is None:
        raise ValueError("YAML config must provide `dataset.name`.")
    if args.model is None:
        raise ValueError("YAML config must provide `model.name`.")

    dataset_class = get_dataset_class(args.dataset)
    declared_dataset_fields = set(_collect_declared_config_fields(dataset_class.config_class))
    seeded_dataset_defaults = {
        key: value
        for key, value in (default_dataset_config or {}).items()
        if key in declared_dataset_fields
    }
    dataset_config = {
        **_default_config_dict(dataset_class.config_class),
        **seeded_dataset_defaults,
        **dataset_values,
    }
    if validate_sections:
        _validate_config_dict(dataset_config, dataset_class.config_class, scope="dataset")

    model_class = get_model_class(args.model)
    model_config = {
        **default_model_config,
        **model_values,
    }
    if validate_sections:
        _validate_config_dict(model_config, model_class.config_class, scope="model")

    encoder_config: dict[str, Any] | None = None
    if args.encoder is not None:
        encoder_class = get_module_class(args.encoder)
        encoder_config = {
            **(default_encoder_config or {}),
            **encoder_values,
        }
        if validate_sections:
            _validate_config_dict(encoder_config, encoder_class.config_class, scope="encoder")
    elif encoder_values:
        raise ValueError("Received encoder settings, but no encoder backbone was selected.")

    decoder_config: dict[str, Any] | None = None
    if args.decoder is not None:
        decoder_class = get_module_class(args.decoder)
        decoder_config = {
            **decoder_values,
        }
        if not decoder_config and args.encoder is not None and args.decoder == args.encoder and encoder_config is not None:
            decoder_config = dict(encoder_config)
        if validate_sections:
            _validate_config_dict(decoder_config, decoder_class.config_class, scope="decoder")
    elif decoder_values:
        raise ValueError("Received decoder settings, but no decoder backbone was selected.")

    trainer_config = {**(default_trainer_config or {})}
    base_training_values: dict[str, Any] = {}
    for key, value in trainer_values.items():
        normalized_key = _normalize_name(key)
        if normalized_key in TRAINING_ARGUMENT_FIELDS:
            base_training_values[normalized_key] = value
        else:
            trainer_config[normalized_key] = value
    if validate_sections and default_trainer_config is not None:
        allowed_trainer_fields = set(default_trainer_config)
        for key in trainer_config:
            if key in allowed_trainer_fields:
                continue
            suggestion = difflib.get_close_matches(key, allowed_trainer_fields, n=1)
            suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
            raise ValueError(f"Unknown trainer option {key!r}.{suffix}")

    for field_name in TRAINING_ARGUMENT_FIELDS:
        if field_name in base_training_values:
            setattr(args, field_name, base_training_values[field_name])

    args.resolved_configs = ResolvedConfigArguments(
        dataset_config=dataset_config,
        model_config=model_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        trainer_config=trainer_config,
    )
    return args
