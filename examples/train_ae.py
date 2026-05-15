"""Train deterministic autoencoder-family models on a named dataset."""

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
from autoencoders import (
    AETrainer,
    AdversarialAutoencoderTrainer,
    AdversarialAutoencoderTrainingArguments,
    load_model,
)


MODEL_CHOICES = ["ae", "dae", "cae", "sae", "topksae", "klsae", "wae", "aae"]
COMMON_MODEL_DEFAULTS = {
    "latent_dim": 16,
    "reconstruction_loss": "mse",
}
MODEL_DEFAULTS = {
    "dae": {
        "noise_type": "gaussian",
        "noise_std": 0.1,
        "masking_ratio": 0.3,
        "apply_noise_in_eval": False,
    },
    "cae": {"contractive_weight": 1e-2},
    "sae": {"sparsity_weight": 1e-3},
    "topksae": {"topk": 4},
    "klsae": {"sparsity_weight": 1e-3, "target_activation": 0.05},
    "wae": {"mmd_weight": 10.0, "mmd_bandwidths": [0.1, 0.2, 0.5, 1.0, 2.0]},
    "aae": {"adversarial_weight": 1.0, "discriminator_hidden_dims": [128, 64]},
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
    parser.add_argument("--model", default="ae", choices=MODEL_CHOICES, help="Model name.")
    parser.add_argument("--generator-learning-rate", type=float, default=None, help="Optional AAE encoder adversarial optimizer learning rate.")
    parser.add_argument("--discriminator-learning-rate", type=float, default=None, help="Optional AAE discriminator optimizer learning rate.")
    parser.add_argument("--discriminator-steps", type=int, default=1, help="Number of AAE discriminator updates per batch.")
    parser.epilog = (
        "Backbone and model options use dotted syntax. "
        "Examples: --model.latent_dim 16 --encoder mlp --encoder.hidden_dims \"[128, 64]\" "
        "--dataset.encoder sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parse_config_arguments(
        parser,
        default_dataset_config=DATASET_DEFAULT_CONFIG,
        default_model_config={**COMMON_MODEL_DEFAULTS, **MODEL_DEFAULTS.get("ae", {})},
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


def build_trainer(args: argparse.Namespace, model):
    if args.model == "aae":
        return AdversarialAutoencoderTrainer(
            model=model,
            args=AdversarialAutoencoderTrainingArguments(
                discriminator_learning_rate=args.discriminator_learning_rate,
                generator_learning_rate=args.generator_learning_rate,
                discriminator_steps=args.discriminator_steps,
                **build_training_arguments(args).__dict__,
            ),
        )
    return AETrainer(model=model, args=build_training_arguments(args))


def main() -> None:
    args = parse_args()
    _, dataloaders, input_dim = prepare_training(args)
    model = build_model(args, input_dim=input_dim)
    print_training_overview(args, model, input_dim=input_dim)
    validate_model_input_compatibility(args, model, dataloaders)
    trainer = build_trainer(args, model)
    trainer.fit(dataloaders, metadata={"dataset": args.dataset, "model": args.model})


if __name__ == "__main__":
    main()
