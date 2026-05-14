"""Train variational autoencoder-family models on a named dataset."""

from __future__ import annotations

import argparse

from _train_common import add_dataset_args, add_training_args, build_training_arguments, prepare_training
from autoencoders import VAETrainer, load_model


MODEL_CHOICES = ["vae", "dvae", "betavae", "hvae", "infovae", "vamppriorvae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    add_training_args(parser)
    parser.add_argument("--model", default="vae", choices=MODEL_CHOICES, help="Model name.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Encoder hidden dims.")
    parser.add_argument("--activation", default="relu", help="Activation name.")
    parser.add_argument("--reconstruction-loss", default="mse", help="Reconstruction loss name.")
    parser.add_argument("--kl-weight", type=float, default=0.1, help="VAE KL loss weight.")
    parser.add_argument("--beta", type=float, default=4.0, help="Beta-VAE KL multiplier.")
    parser.add_argument("--top-latent-dim", type=int, default=None, help="Hierarchical VAE top-level latent dimensionality.")
    parser.add_argument("--mmd-weight", type=float, default=5.0, help="InfoVAE MMD prior-matching weight.")
    parser.add_argument("--mmd-bandwidths", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0], help="InfoVAE MMD kernel bandwidths.")
    parser.add_argument("--num-pseudo-inputs", type=int, default=128, help="VampPriorVAE number of learned pseudo-inputs.")
    parser.add_argument("--pseudo-input-std", type=float, default=0.01, help="VampPriorVAE pseudo-input initialization std.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=20, help="Number of epochs for VAE KL warmup.")
    parser.add_argument("--kl-start-weight", type=float, default=0.0, help="Starting KL weight during warmup.")
    parser.add_argument("--free-bits", type=float, default=0.02, help="Per-latent-dimension free bits floor for VAE KL.")
    parser.add_argument("--noise-type", default="gaussian", help="DVAE noise type.")
    parser.add_argument("--noise-std", type=float, default=0.1, help="DVAE gaussian noise std.")
    parser.add_argument("--masking-ratio", type=float, default=0.3, help="DVAE masking ratio.")
    parser.add_argument(
        "--apply-noise-in-eval",
        action="store_true",
        help="Whether denoising variational autoencoders should also corrupt inputs during evaluation.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace, input_dim: int):
    model_kwargs = {
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "reconstruction_loss": args.reconstruction_loss,
        "free_bits": args.free_bits,
        "kl_warmup_epochs": args.kl_warmup_epochs,
        "kl_start_weight": args.kl_start_weight,
    }
    if args.model == "betavae":
        model_kwargs.update(beta=args.beta)
    else:
        model_kwargs.update(kl_weight=args.kl_weight)
    if args.model == "infovae":
        model_kwargs.update(
            mmd_weight=args.mmd_weight,
            mmd_bandwidths=list(args.mmd_bandwidths),
        )
    if args.model == "vamppriorvae":
        model_kwargs.update(
            num_pseudo_inputs=args.num_pseudo_inputs,
            pseudo_input_std=args.pseudo_input_std,
        )
    if args.model == "dvae":
        model_kwargs.update(
            noise_type=args.noise_type,
            noise_std=args.noise_std,
            masking_ratio=args.masking_ratio,
            apply_noise_in_eval=args.apply_noise_in_eval,
        )
    if args.model == "hvae":
        model_kwargs.update(top_latent_dim=args.top_latent_dim)
    return load_model(args.model, **model_kwargs)


def main() -> None:
    args = parse_args()
    _, dataloaders, input_dim = prepare_training(args)
    model = build_model(args, input_dim=input_dim)
    trainer = VAETrainer(model=model, args=build_training_arguments(args))
    trainer.fit(dataloaders, metadata={"dataset": args.dataset, "model": args.model})


if __name__ == "__main__":
    main()
