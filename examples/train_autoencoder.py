"""Train and evaluate an autoencoder-family model on a named dataset."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders import (
    AdversarialAutoencoderTrainer,
    AdversarialAutoencoderTrainingArguments,
    AutoencoderTrainer,
    ContractiveAutoencoderTrainer,
    QuantizedAutoencoderTrainer,
    QuantizedAutoencoderTrainingArguments,
    TrainingArguments,
    VAETrainer,
    VAETrainingArguments,
    load_dataset,
    load_model,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="glove", choices=["glove", "fasttext", "numberbatch"], help="Dataset name.")
    parser.add_argument(
        "--model",
        default="ae",
        choices=["ae", "dae", "cae", "sae", "topksae", "klsae", "vae", "dvae", "betavae", "hvae", "wae", "aae", "vqvae", "fsq", "pqvae", "rqvae"],
        help="Model name.",
    )
    parser.add_argument("--output-dir", default="artifacts/train-autoencoder", help="Model output directory.")

    parser.add_argument("--dim", type=int, default=50, help="Dataset embedding dimension when supported.")
    parser.add_argument("--max-vectors", type=int, default=None, help="Optional dataset cap for faster experiments.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting and training.")

    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Encoder hidden dims.")
    parser.add_argument("--activation", default="relu", help="Activation name.")
    parser.add_argument("--reconstruction-loss", default="mse", help="Reconstruction loss name.")
    parser.add_argument("--contractive-weight", type=float, default=1e-2, help="Contractive-AE Jacobian penalty weight.")
    parser.add_argument("--sparsity-weight", type=float, default=1e-3, help="Sparse-AE latent L1 regularization weight.")
    parser.add_argument("--topk", type=int, default=4, help="TopK-SAE number of active latent units per sample.")
    parser.add_argument("--target-activation", type=float, default=0.05, help="KL-SAE target latent activation probability.")
    parser.add_argument("--kl-weight", type=float, default=0.1, help="VAE KL loss weight.")
    parser.add_argument("--beta", type=float, default=4.0, help="Beta-VAE KL multiplier.")
    parser.add_argument("--top-latent-dim", type=int, default=None, help="Hierarchical VAE top-level latent dimensionality.")
    parser.add_argument("--mmd-weight", type=float, default=10.0, help="WAE MMD regularization weight.")
    parser.add_argument("--mmd-bandwidths", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0, 2.0], help="WAE MMD kernel bandwidths.")
    parser.add_argument("--adversarial-weight", type=float, default=1.0, help="AAE adversarial regularization weight.")
    parser.add_argument("--discriminator-hidden-dims", type=int, nargs="+", default=[128, 64], help="AAE discriminator hidden dims.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=20, help="Number of epochs for VAE KL warmup.")
    parser.add_argument("--kl-start-weight", type=float, default=0.0, help="Starting KL weight during warmup.")
    parser.add_argument("--free-bits", type=float, default=0.02, help="Per-latent-dimension free bits floor for VAE KL.")
    parser.add_argument("--codebook-size", type=int, default=256, help="VQ-VAE codebook size.")
    parser.add_argument("--num-levels", type=int, default=8, help="FSQ number of scalar quantization levels.")
    parser.add_argument("--num-codebooks", type=int, default=2, help="PQ-VAE number of product codebooks.")
    parser.add_argument("--num-quantizers", type=int, default=2, help="RQ-VAE number of residual quantizers.")
    parser.add_argument("--commitment-weight", type=float, default=0.25, help="VQ-VAE commitment loss weight.")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="VQ-VAE codebook loss weight.")
    parser.add_argument(
        "--use-ema-codebook",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether VQ-VAE should update the codebook with EMA instead of gradient-based codebook loss.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay for VQ-VAE codebook updates.")
    parser.add_argument("--ema-epsilon", type=float, default=1e-5, help="Numerical stability epsilon for EMA codebook updates.")
    parser.add_argument(
        "--dead-code-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether QuantizedAutoencoderTrainer should reset dead VQ codes after training epochs.",
    )
    parser.add_argument("--dead-code-threshold", type=int, default=0, help="Reset VQ codes whose epoch usage count is at or below this threshold.")

    parser.add_argument("--noise-type", default="gaussian", help="DAE noise type.")
    parser.add_argument("--noise-std", type=float, default=0.1, help="DAE gaussian noise std.")
    parser.add_argument("--masking-ratio", type=float, default=0.3, help="DAE masking ratio.")
    parser.add_argument(
        "--apply-noise-in-eval",
        action="store_true",
        help="Whether denoising autoencoders should also corrupt inputs during evaluation.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs. Set to 0 to train until early stopping triggers.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs without validation improvement.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--generator-learning-rate", type=float, default=None, help="Optional AAE encoder adversarial optimizer learning rate.")
    parser.add_argument("--discriminator-learning-rate", type=float, default=None, help="Optional AAE discriminator optimizer learning rate.")
    parser.add_argument("--discriminator-steps", type=int, default=1, help="Number of AAE discriminator updates per batch.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, mps.")
    parser.add_argument(
        "--show-only-best-epochs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to persist only best-epoch summaries in the terminal output.",
    )
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    dim = args.dim
    if args.dataset == "fasttext" and dim == 50:
        dim = 300
    if args.dataset == "numberbatch" and dim == 50:
        dim = 300
    dataset_kwargs = {
        "dim": dim,
        "max_vectors": args.max_vectors,
    }
    return load_dataset(args.dataset, **dataset_kwargs)


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
            {
                "noise_type": args.noise_type,
                "noise_std": args.noise_std,
                "masking_ratio": args.masking_ratio,
                "apply_noise_in_eval": args.apply_noise_in_eval,
            }
        )
    if args.model == "cae":
        model_kwargs.update(
            {
                "contractive_weight": args.contractive_weight,
            }
        )
    if args.model == "sae":
        model_kwargs.update(
            {
                "sparsity_weight": args.sparsity_weight,
            }
        )
    if args.model == "topksae":
        model_kwargs.update(
            {
                "topk": args.topk,
            }
        )
    if args.model == "klsae":
        model_kwargs.update(
            {
                "sparsity_weight": args.sparsity_weight,
                "target_activation": args.target_activation,
            }
        )
    if args.model == "vae":
        model_kwargs.update(
            {
                "kl_weight": args.kl_weight,
                "free_bits": args.free_bits,
                "kl_warmup_epochs": args.kl_warmup_epochs,
                "kl_start_weight": args.kl_start_weight,
            }
        )
    if args.model == "dvae":
        model_kwargs.update(
            {
                "kl_weight": args.kl_weight,
                "free_bits": args.free_bits,
                "kl_warmup_epochs": args.kl_warmup_epochs,
                "kl_start_weight": args.kl_start_weight,
                "noise_type": args.noise_type,
                "noise_std": args.noise_std,
                "masking_ratio": args.masking_ratio,
                "apply_noise_in_eval": args.apply_noise_in_eval,
            }
        )
    if args.model == "betavae":
        model_kwargs.update(
            {
                "beta": args.beta,
                "free_bits": args.free_bits,
                "kl_warmup_epochs": args.kl_warmup_epochs,
                "kl_start_weight": args.kl_start_weight,
            }
        )
    if args.model == "hvae":
        model_kwargs.update(
            {
                "kl_weight": args.kl_weight,
                "free_bits": args.free_bits,
                "kl_warmup_epochs": args.kl_warmup_epochs,
                "kl_start_weight": args.kl_start_weight,
                "top_latent_dim": args.top_latent_dim,
            }
        )
    if args.model == "wae":
        model_kwargs.update(
            {
                "mmd_weight": args.mmd_weight,
                "mmd_bandwidths": list(args.mmd_bandwidths),
            }
        )
    if args.model == "aae":
        model_kwargs.update(
            {
                "adversarial_weight": args.adversarial_weight,
                "discriminator_hidden_dims": list(args.discriminator_hidden_dims),
            }
        )
    if args.model == "vqvae":
        model_kwargs.update(
            {
                "codebook_size": args.codebook_size,
                "commitment_weight": args.commitment_weight,
                "codebook_weight": args.codebook_weight,
                "use_ema_codebook": args.use_ema_codebook,
                "ema_decay": args.ema_decay,
                "ema_epsilon": args.ema_epsilon,
            }
        )
    if args.model == "fsq":
        model_kwargs.update(
            {
                "num_levels": args.num_levels,
                "commitment_weight": args.commitment_weight,
            }
        )
    if args.model == "pqvae":
        model_kwargs.update(
            {
                "codebook_size": args.codebook_size,
                "num_codebooks": args.num_codebooks,
                "commitment_weight": args.commitment_weight,
                "codebook_weight": args.codebook_weight,
                "use_ema_codebook": args.use_ema_codebook,
                "ema_decay": args.ema_decay,
                "ema_epsilon": args.ema_epsilon,
            }
        )
    if args.model == "rqvae":
        model_kwargs.update(
            {
                "codebook_size": args.codebook_size,
                "num_quantizers": args.num_quantizers,
                "commitment_weight": args.commitment_weight,
                "codebook_weight": args.codebook_weight,
                "use_ema_codebook": args.use_ema_codebook,
                "ema_decay": args.ema_decay,
                "ema_epsilon": args.ema_epsilon,
            }
        )

    return load_model(args.model, **model_kwargs)


def build_trainer(args: argparse.Namespace, model):
    common_kwargs = {
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "show_only_best_epochs": args.show_only_best_epochs,
    }

    if args.model == "vae" or args.model == "dvae" or args.model == "betavae" or args.model == "hvae":
        training_args = VAETrainingArguments(**common_kwargs)
        return VAETrainer(model=model, args=training_args)
    if args.model == "cae":
        training_args = TrainingArguments(**common_kwargs)
        return ContractiveAutoencoderTrainer(model=model, args=training_args)
    if args.model == "aae":
        training_args = AdversarialAutoencoderTrainingArguments(
            discriminator_learning_rate=args.discriminator_learning_rate,
            generator_learning_rate=args.generator_learning_rate,
            discriminator_steps=args.discriminator_steps,
            **common_kwargs,
        )
        return AdversarialAutoencoderTrainer(model=model, args=training_args)
    if args.model == "vqvae" or args.model == "fsq" or args.model == "pqvae" or args.model == "rqvae":
        training_args = QuantizedAutoencoderTrainingArguments(
            dead_code_reset=args.dead_code_reset,
            dead_code_threshold=args.dead_code_threshold,
            **common_kwargs,
        )
        return QuantizedAutoencoderTrainer(model=model, args=training_args)

    training_args = TrainingArguments(**common_kwargs)
    return AutoencoderTrainer(model=model, args=training_args)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = build_dataset(args)
    dataloaders = dataset.get_dataloaders(
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    embedding_matrix = dataset.load_embedding_matrix()
    model = build_model(args, input_dim=embedding_matrix.embedding_dim)
    trainer = build_trainer(args, model)
    trainer.fit(
        dataloaders,
        metadata={
            "dataset": args.dataset,
            "model": args.model,
        },
    )



if __name__ == "__main__":
    main()
