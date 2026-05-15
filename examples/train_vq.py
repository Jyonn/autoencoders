"""Train quantized autoencoder-family models on a named dataset."""

from __future__ import annotations

import argparse

from _train_common import (
    add_dataset_args,
    add_training_args,
    build_mlp_backbone_kwargs,
    build_training_arguments,
    prepare_training,
    print_training_overview,
    validate_model_input_compatibility,
)
from autoencoders import VQTrainer, load_model


MODEL_CHOICES = ["vqvae", "fsq", "pqvae", "rqvae", "gumbelvq", "rfsq", "vqvae2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    add_training_args(parser)
    parser.add_argument("--model", default="vqvae", choices=MODEL_CHOICES, help="Model name.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Encoder hidden dims.")
    parser.add_argument("--activation", default="relu", help="Activation name.")
    parser.add_argument("--reconstruction-loss", default="mse", help="Reconstruction loss name.")
    parser.add_argument("--codebook-size", type=int, default=256, help="Codebook size.")
    parser.add_argument("--num-levels", type=int, default=8, help="FSQ number of scalar quantization levels.")
    parser.add_argument("--num-codebooks", type=int, default=2, help="PQ-VAE number of product codebooks.")
    parser.add_argument("--num-quantizers", type=int, default=2, help="RQ-VAE number of residual quantizers.")
    parser.add_argument("--top-latent-dim", type=int, default=None, help="VQ-VAE-2 top-level latent width.")
    parser.add_argument("--commitment-weight", type=float, default=0.25, help="Commitment loss weight.")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="Codebook loss weight.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Gumbel-VQ softmax temperature.")
    parser.add_argument(
        "--straight-through",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether Gumbel-VQ uses hard straight-through assignments.",
    )
    parser.add_argument(
        "--use-ema-codebook",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the codebook should update with EMA instead of gradient-based codebook loss.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay for codebook updates.")
    parser.add_argument("--ema-epsilon", type=float, default=1e-5, help="Numerical stability epsilon for EMA codebook updates.")
    parser.add_argument(
        "--dead-code-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether VQ-family models should reset dead codes at the end of each training epoch.",
    )
    parser.add_argument("--dead-code-threshold", type=int, default=0, help="Reset codes whose epoch usage count is at or below this threshold.")
    return parser.parse_args()


def build_model(args: argparse.Namespace, input_dim: int):
    model_kwargs = {
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "activation": args.activation,
        "reconstruction_loss": args.reconstruction_loss,
        **build_mlp_backbone_kwargs(args.hidden_dims, args.activation),
    }
    if args.model == "fsq":
        model_kwargs.update(num_levels=args.num_levels, commitment_weight=args.commitment_weight)
    elif args.model == "rfsq":
        model_kwargs.update(
            num_levels=args.num_levels,
            commitment_weight=args.commitment_weight,
            num_quantizers=args.num_quantizers,
        )
    elif args.model == "gumbelvq":
        model_kwargs.update(
            codebook_size=args.codebook_size,
            commitment_weight=args.commitment_weight,
            codebook_weight=args.codebook_weight,
            dead_code_reset=args.dead_code_reset,
            dead_code_threshold=args.dead_code_threshold,
            temperature=args.temperature,
            straight_through=args.straight_through,
        )
    else:
        model_kwargs.update(
            codebook_size=args.codebook_size,
            commitment_weight=args.commitment_weight,
            codebook_weight=args.codebook_weight,
            use_ema_codebook=args.use_ema_codebook,
            ema_decay=args.ema_decay,
            ema_epsilon=args.ema_epsilon,
            dead_code_reset=args.dead_code_reset,
            dead_code_threshold=args.dead_code_threshold,
        )
        if args.model == "pqvae":
            model_kwargs.update(num_codebooks=args.num_codebooks)
        if args.model == "rqvae":
            model_kwargs.update(num_quantizers=args.num_quantizers)
        if args.model == "vqvae2":
            model_kwargs.update(top_latent_dim=args.top_latent_dim)
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
