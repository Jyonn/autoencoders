"""Train deterministic autoencoder-family models on a named dataset."""

from __future__ import annotations

import argparse

from _train_common import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    add_training_args(parser)
    parser.add_argument("--model", default="ae", choices=MODEL_CHOICES, help="Model name.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Encoder hidden dims.")
    parser.add_argument("--activation", default="relu", help="Activation name.")
    parser.add_argument("--reconstruction-loss", default="mse", help="Reconstruction loss name.")
    parser.add_argument("--contractive-weight", type=float, default=1e-2, help="Contractive-AE Jacobian penalty weight.")
    parser.add_argument("--sparsity-weight", type=float, default=1e-3, help="Sparse-AE latent L1 regularization weight.")
    parser.add_argument("--topk", type=int, default=4, help="TopK-SAE number of active latent units per sample.")
    parser.add_argument("--target-activation", type=float, default=0.05, help="KL-SAE target latent activation probability.")
    parser.add_argument("--mmd-weight", type=float, default=10.0, help="WAE MMD regularization weight.")
    parser.add_argument("--mmd-bandwidths", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0, 2.0], help="WAE MMD kernel bandwidths.")
    parser.add_argument("--adversarial-weight", type=float, default=1.0, help="AAE adversarial regularization weight.")
    parser.add_argument("--discriminator-hidden-dims", type=int, nargs="+", default=[128, 64], help="AAE discriminator hidden dims.")
    parser.add_argument("--generator-learning-rate", type=float, default=None, help="Optional AAE encoder adversarial optimizer learning rate.")
    parser.add_argument("--discriminator-learning-rate", type=float, default=None, help="Optional AAE discriminator optimizer learning rate.")
    parser.add_argument("--discriminator-steps", type=int, default=1, help="Number of AAE discriminator updates per batch.")
    parser.add_argument("--noise-type", default="gaussian", help="DAE noise type.")
    parser.add_argument("--noise-std", type=float, default=0.1, help="DAE gaussian noise std.")
    parser.add_argument("--masking-ratio", type=float, default=0.3, help="DAE masking ratio.")
    parser.add_argument(
        "--apply-noise-in-eval",
        action="store_true",
        help="Whether denoising autoencoders should also corrupt inputs during evaluation.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace, input_dim: int):
    model_kwargs = {
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "reconstruction_loss": args.reconstruction_loss,
    }
    if args.model == "dae":
        model_kwargs.update(
            noise_type=args.noise_type,
            noise_std=args.noise_std,
            masking_ratio=args.masking_ratio,
            apply_noise_in_eval=args.apply_noise_in_eval,
        )
    if args.model == "cae":
        model_kwargs.update(contractive_weight=args.contractive_weight)
    if args.model == "sae":
        model_kwargs.update(sparsity_weight=args.sparsity_weight)
    if args.model == "topksae":
        model_kwargs.update(topk=args.topk)
    if args.model == "klsae":
        model_kwargs.update(sparsity_weight=args.sparsity_weight, target_activation=args.target_activation)
    if args.model == "wae":
        model_kwargs.update(mmd_weight=args.mmd_weight, mmd_bandwidths=list(args.mmd_bandwidths))
    if args.model == "aae":
        model_kwargs.update(
            adversarial_weight=args.adversarial_weight,
            discriminator_hidden_dims=list(args.discriminator_hidden_dims),
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
