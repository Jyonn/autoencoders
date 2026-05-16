"""Train variational autoencoder-family models on a named dataset."""

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
    FactorVAETrainer,
    FactorVariationalAutoencoderTrainingArguments,
    VAETrainer,
    load_model,
)


MODEL_CHOICES = ["vae", "dvae", "betavae", "hvae", "infovae", "mmdvae", "vamppriorvae", "factorvae", "dipvae", "betatcvae"]
COMMON_MODEL_DEFAULTS = {
    "latent_dim": 16,
    "reconstruction_loss": "mse",
}
MODEL_DEFAULTS = {
    "vae": {"kl_weight": 0.1, "free_bits": 0.02, "kl_warmup_epochs": 20, "kl_start_weight": 0.0},
    "dvae": {
        "kl_weight": 0.1,
        "free_bits": 0.02,
        "kl_warmup_epochs": 20,
        "kl_start_weight": 0.0,
        "noise_type": "gaussian",
        "noise_std": 0.1,
        "masking_ratio": 0.3,
        "apply_noise_in_eval": False,
    },
    "betavae": {"beta": 4.0, "free_bits": 0.02, "kl_warmup_epochs": 20, "kl_start_weight": 0.0},
    "hvae": {"kl_weight": 0.1, "free_bits": 0.02, "kl_warmup_epochs": 20, "kl_start_weight": 0.0},
    "infovae": {
        "kl_weight": 0.1,
        "free_bits": 0.02,
        "kl_warmup_epochs": 20,
        "kl_start_weight": 0.0,
        "mmd_weight": 5.0,
        "mmd_bandwidths": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    },
    "mmdvae": {
        "free_bits": 0.02,
        "mmd_weight": 5.0,
        "mmd_bandwidths": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    },
    "vamppriorvae": {
        "kl_weight": 0.1,
        "free_bits": 0.02,
        "kl_warmup_epochs": 20,
        "kl_start_weight": 0.0,
        "num_pseudo_inputs": 128,
        "pseudo_input_std": 0.01,
    },
    "factorvae": {
        "kl_weight": 0.1,
        "free_bits": 0.02,
        "kl_warmup_epochs": 20,
        "kl_start_weight": 0.0,
        "tc_weight": 10.0,
        "discriminator_hidden_dims": [128, 64],
    },
    "dipvae": {
        "kl_weight": 0.1,
        "free_bits": 0.02,
        "kl_warmup_epochs": 20,
        "kl_start_weight": 0.0,
        "dip_weight": 10.0,
        "dip_offdiag_weight": 1.0,
        "dip_diag_weight": 1.0,
    },
    "betatcvae": {
        "free_bits": 0.02,
        "kl_warmup_epochs": 20,
        "kl_start_weight": 0.0,
        "mutual_information_weight": 1.0,
        "total_correlation_weight": 6.0,
        "dimension_wise_kl_weight": 1.0,
    },
}
DEFAULT_TRAINER_CONFIG = {
    "discriminator_learning_rate": None,
    "discriminator_steps": 1,
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
    parser.add_argument("--model", default="vae", choices=MODEL_CHOICES, help="Model name.")
    parser.epilog = (
        "Model and backbone options use dotted syntax. "
        "Examples: --model.kl_weight 0.1 --encoder mlp --encoder.hidden_dims \"[128, 64]\" "
        "--trainer.discriminator_steps 2 "
        "--dataset.encoder sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parse_config_arguments(
        parser,
        default_dataset_config=DATASET_DEFAULT_CONFIG,
        default_trainer_config=DEFAULT_TRAINER_CONFIG,
        default_model_config={**COMMON_MODEL_DEFAULTS, **MODEL_DEFAULTS.get("vae", {})},
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
    if args.model == "factorvae":
        base_args = build_training_arguments(args)
        trainer_config = args.resolved_configs.trainer_config
        trainer_args = FactorVariationalAutoencoderTrainingArguments(
            output_dir=base_args.output_dir,
            epochs=base_args.epochs,
            patience=base_args.patience,
            learning_rate=base_args.learning_rate,
            batch_size=base_args.batch_size,
            device=base_args.device,
            seed=base_args.seed,
            show_only_best_epochs=base_args.show_only_best_epochs,
            discriminator_learning_rate=trainer_config["discriminator_learning_rate"],
            discriminator_steps=trainer_config["discriminator_steps"],
        )
        trainer = FactorVAETrainer(model=model, args=trainer_args)
    else:
        trainer = VAETrainer(model=model, args=build_training_arguments(args))
    trainer.fit(dataloaders, metadata={"dataset": args.dataset, "model": args.model})


if __name__ == "__main__":
    main()
