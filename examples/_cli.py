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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders.models.loading import get_model_class
from autoencoders.modules.loading import get_module_class


LEGACY_FLAG_SPECS: dict[str, tuple[str, str]] = {
    "--dataset-encoder": ("dataset.encoder", "value"),
    "--dataset-encoder-batch-size": ("dataset.encoder_batch_size", "value"),
    "--dataset-clip-pretrained": ("dataset.clip_pretrained", "value"),
    "--dataset-clip-device": ("dataset.clip_device", "value"),
    "--dataset-clip-modality": ("dataset.clip_modality", "value"),
    "--latent-dim": ("model.latent_dim", "value"),
    "--reconstruction-loss": ("model.reconstruction_loss", "value"),
    "--contractive-weight": ("model.contractive_weight", "value"),
    "--sparsity-weight": ("model.sparsity_weight", "value"),
    "--topk": ("model.topk", "value"),
    "--target-activation": ("model.target_activation", "value"),
    "--mmd-weight": ("model.mmd_weight", "value"),
    "--mmd-bandwidths": ("model.mmd_bandwidths", "many"),
    "--adversarial-weight": ("model.adversarial_weight", "value"),
    "--discriminator-hidden-dims": ("model.discriminator_hidden_dims", "many"),
    "--generator-learning-rate": ("model.generator_learning_rate", "value"),
    "--discriminator-learning-rate": ("model.discriminator_learning_rate", "value"),
    "--discriminator-steps": ("model.discriminator_steps", "value"),
    "--noise-type": ("model.noise_type", "value"),
    "--noise-std": ("model.noise_std", "value"),
    "--masking-ratio": ("model.masking_ratio", "value"),
    "--apply-noise-in-eval": ("model.apply_noise_in_eval", "flag_true"),
    "--no-apply-noise-in-eval": ("model.apply_noise_in_eval", "flag_false"),
    "--kl-weight": ("model.kl_weight", "value"),
    "--beta": ("model.beta", "value"),
    "--top-latent-dim": ("model.top_latent_dim", "value"),
    "--num-pseudo-inputs": ("model.num_pseudo_inputs", "value"),
    "--pseudo-input-std": ("model.pseudo_input_std", "value"),
    "--tc-weight": ("model.tc_weight", "value"),
    "--mutual-information-weight": ("model.mutual_information_weight", "value"),
    "--total-correlation-weight": ("model.total_correlation_weight", "value"),
    "--dimension-wise-kl-weight": ("model.dimension_wise_kl_weight", "value"),
    "--dip-weight": ("model.dip_weight", "value"),
    "--dip-offdiag-weight": ("model.dip_offdiag_weight", "value"),
    "--dip-diag-weight": ("model.dip_diag_weight", "value"),
    "--kl-warmup-epochs": ("model.kl_warmup_epochs", "value"),
    "--kl-start-weight": ("model.kl_start_weight", "value"),
    "--free-bits": ("model.free_bits", "value"),
    "--codebook-size": ("model.codebook_size", "value"),
    "--num-levels": ("model.num_levels", "value"),
    "--num-codebooks": ("model.num_codebooks", "value"),
    "--num-quantizers": ("model.num_quantizers", "value"),
    "--commitment-weight": ("model.commitment_weight", "value"),
    "--codebook-weight": ("model.codebook_weight", "value"),
    "--temperature": ("model.temperature", "value"),
    "--straight-through": ("model.straight_through", "flag_true"),
    "--no-straight-through": ("model.straight_through", "flag_false"),
    "--use-ema-codebook": ("model.use_ema_codebook", "flag_true"),
    "--no-use-ema-codebook": ("model.use_ema_codebook", "flag_false"),
    "--ema-decay": ("model.ema_decay", "value"),
    "--ema-epsilon": ("model.ema_epsilon", "value"),
    "--dead-code-reset": ("model.dead_code_reset", "flag_true"),
    "--no-dead-code-reset": ("model.dead_code_reset", "flag_false"),
    "--dead-code-threshold": ("model.dead_code_threshold", "value"),
    "--hidden-dims": ("encoder.hidden_dims", "many"),
    "--activation": ("encoder.activation", "value"),
    "--use-bias": ("encoder.use_bias", "flag_true"),
    "--no-use-bias": ("encoder.use_bias", "flag_false"),
}


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


def _serialize_cli_value(value: Any) -> str:
    return json.dumps(value)


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


def _normalize_legacy_tokens(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0

    while index < len(argv):
        token = argv[index]
        spec = LEGACY_FLAG_SPECS.get(token)
        if spec is None:
            normalized.append(token)
            index += 1
            continue

        dotted_name, mode = spec
        dotted_flag = f"--{dotted_name}"
        if mode == "flag_true":
            normalized.extend([dotted_flag, "true"])
            index += 1
            continue
        if mode == "flag_false":
            normalized.extend([dotted_flag, "false"])
            index += 1
            continue
        if index + 1 >= len(argv):
            raise ValueError(f"Expected a value after {token!r}.")
        if mode == "many":
            values: list[Any] = []
            next_index = index + 1
            while next_index < len(argv) and not argv[next_index].startswith("--"):
                values.append(_parse_cli_value(argv[next_index]))
                next_index += 1
            if not values:
                raise ValueError(f"Expected one or more values after {token!r}.")
            normalized.extend([dotted_flag, _serialize_cli_value(values)])
            index = next_index
            continue
        normalized.extend([dotted_flag, argv[index + 1]])
        index += 2

    return normalized


def _parse_dotted_overrides(tokens: list[str]) -> dict[str, dict[str, Any]]:
    overrides = {"dataset": {}, "model": {}, "encoder": {}, "decoder": {}}
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
    ) -> None:
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config


def parse_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_dataset_config: dict[str, Any] | None = None,
    default_model_config: dict[str, Any],
    default_encoder: str | None,
    default_encoder_config: dict[str, Any] | None,
) -> argparse.Namespace:
    normalized_argv = _normalize_legacy_tokens(sys.argv[1:])
    args, unknown = parser.parse_known_args(normalized_argv)
    overrides = _parse_dotted_overrides(unknown)

    dataset_config = {**(default_dataset_config or {}), **overrides["dataset"]}
    if default_dataset_config is not None:
        allowed_dataset_fields = set(default_dataset_config)
        for key in dataset_config:
            if key in allowed_dataset_fields:
                continue
            suggestion = difflib.get_close_matches(key, allowed_dataset_fields, n=1)
            suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
            raise ValueError(f"Unknown dataset option {key!r}.{suffix}")

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
    )
    return args
