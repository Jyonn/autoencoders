"""Train quantized autoencoder-family models on a named dataset."""

from __future__ import annotations

import argparse

from _cli import parse_config_arguments
from _train_common import (
    DATASET_DEFAULT_CONFIG,
    add_backbone_args,
    add_dataset_args,
    add_training_args,
    build_training_arguments,
    prepare_training,
    print_training_overview,
    validate_model_input_compatibility,
)
from autoencoders import VQTrainer, load_model


MODEL_CHOICES = ["vqvae", "fsq", "pqvae", "rqvae", "gumbelvq", "rfsq", "vqvae2"]
COMMON_MODEL_DEFAULTS = {
    "latent_dim": 16,
    "reconstruction_loss": "mse",
}
MODEL_DEFAULTS = {
    "vqvae": {
        "codebook_size": 256,
        "commitment_weight": 0.25,
        "codebook_weight": 1.0,
        "use_ema_codebook": False,
        "ema_decay": 0.99,
        "ema_epsilon": 1e-5,
        "dead_code_reset": True,
        "dead_code_threshold": 0,
    },
    "gumbelvq": {
        "codebook_size": 256,
        "commitment_weight": 0.25,
        "codebook_weight": 1.0,
        "dead_code_reset": True,
        "dead_code_threshold": 0,
        "temperature": 1.0,
        "straight_through": True,
    },
    "fsq": {"num_levels": 8, "commitment_weight": 0.25},
    "rfsq": {"num_levels": 8, "commitment_weight": 0.25, "num_quantizers": 2},
    "pqvae": {
        "codebook_size": 256,
        "num_codebooks": 2,
        "commitment_weight": 0.25,
        "codebook_weight": 1.0,
        "use_ema_codebook": False,
        "ema_decay": 0.99,
        "ema_epsilon": 1e-5,
        "dead_code_reset": True,
        "dead_code_threshold": 0,
    },
    "rqvae": {
        "codebook_size": 256,
        "num_quantizers": 2,
        "commitment_weight": 0.25,
        "codebook_weight": 1.0,
        "use_ema_codebook": False,
        "ema_decay": 0.99,
        "ema_epsilon": 1e-5,
        "dead_code_reset": True,
        "dead_code_threshold": 0,
    },
    "vqvae2": {
        "codebook_size": 256,
        "commitment_weight": 0.25,
        "codebook_weight": 1.0,
        "use_ema_codebook": False,
        "ema_decay": 0.99,
        "ema_epsilon": 1e-5,
        "dead_code_reset": True,
        "dead_code_threshold": 0,
    },
}
DEFAULT_ENCODER_CONFIG = {
    "hidden_dims": [64, 32],
    "activation": "relu",
    "use_bias": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    add_training_args(parser)
    add_backbone_args(parser, default_encoder="mlp")
    parser.add_argument("--model", default="vqvae", choices=MODEL_CHOICES, help="Model name.")
    parser.epilog = (
        "Model and backbone options use dotted syntax. "
        "Examples: --model.codebook_size 256 --encoder mlp --encoder.hidden_dims \"[128, 64]\" "
        "--dataset.encoder ViT-B-32"
    )
    args = parse_config_arguments(
        parser,
        default_dataset_config=DATASET_DEFAULT_CONFIG,
        default_model_config={**COMMON_MODEL_DEFAULTS, **MODEL_DEFAULTS.get("vqvae", {})},
        default_encoder="mlp",
        default_encoder_config=DEFAULT_ENCODER_CONFIG,
    )
    args.resolved_configs.model_config = {
        **COMMON_MODEL_DEFAULTS,
        **MODEL_DEFAULTS.get(args.model, {}),
        **args.resolved_configs.model_config,
    }
    return args


def build_model(args: argparse.Namespace, input_dim: int):
    resolved = args.resolved_configs
    model_kwargs = {
        "input_dim": input_dim,
        **resolved.model_config,
        "encoder": args.encoder,
        "encoder_config": resolved.encoder_config,
    }
    if args.decoder is not None:
        model_kwargs.update(
            decoder=args.decoder,
            decoder_config=resolved.decoder_config,
        )
    return load_model(args.model, **model_kwargs)


def main() -> None:
    args = parse_args()
    _, dataloaders, input_dim = prepare_training(args)
    model = build_model(args, input_dim=input_dim)
    print_training_overview(args, model, input_dim=input_dim)
    validate_model_input_compatibility(args, model, dataloaders)
    trainer = VQTrainer(model=model, args=build_training_arguments(args))
    trainer.fit(dataloaders, metadata={"dataset": args.dataset, "model": args.model})


if __name__ == "__main__":
    main()
