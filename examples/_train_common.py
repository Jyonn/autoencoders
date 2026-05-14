"""Shared helpers for training entrypoints."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoencoders import TrainingArguments, load_dataset, set_seed
from autoencoders.training.display import style


DATASET_CHOICES = ["glove", "fasttext", "numberbatch", "snli", "multinli", "flickr30k"]

COMMON_MODEL_PARAMETERS = [
    ("latent_dim", "Latent vector width after encoding."),
    ("hidden_dims", "Hidden layer sizes used by the encoder backbone."),
    ("activation", "Non-linearity used between MLP layers."),
    ("reconstruction_loss", "Objective used to compare inputs and reconstructions."),
]

MODEL_PARAMETER_SECTIONS = {
    "dae": [
        ("noise_type", "How inputs are corrupted before reconstruction."),
        ("noise_std", "Standard deviation for Gaussian corruption."),
        ("masking_ratio", "Fraction of features masked for masking corruption."),
        ("apply_noise_in_eval", "Whether to also corrupt validation and test inputs."),
    ],
    "cae": [
        ("contractive_weight", "Strength of the Jacobian contraction penalty."),
    ],
    "sae": [
        ("sparsity_weight", "Strength of the latent sparsity penalty."),
    ],
    "topksae": [
        ("topk", "Number of latent units allowed to stay active per sample."),
    ],
    "klsae": [
        ("sparsity_weight", "Strength of the KL sparsity regularizer."),
        ("target_activation", "Desired average activation probability per latent unit."),
    ],
    "wae": [
        ("mmd_weight", "Weight of the MMD prior-matching penalty."),
        ("mmd_bandwidths", "Kernel bandwidths used by the MMD estimator."),
    ],
    "aae": [
        ("adversarial_weight", "Strength of the latent adversarial matching loss."),
        ("discriminator_hidden_dims", "Hidden layer sizes of the latent discriminator."),
        ("generator_learning_rate", "Optional optimizer rate for the encoder adversarial path."),
        ("discriminator_learning_rate", "Optional optimizer rate for the discriminator."),
        ("discriminator_steps", "How many discriminator updates run per training batch."),
    ],
    "vae": [
        ("kl_weight", "Weight applied to KL regularization."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
    ],
    "dvae": [
        ("kl_weight", "Weight applied to KL regularization."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
        ("noise_type", "How inputs are corrupted before variational reconstruction."),
        ("noise_std", "Standard deviation for Gaussian corruption."),
        ("masking_ratio", "Fraction of features masked for masking corruption."),
        ("apply_noise_in_eval", "Whether to also corrupt validation and test inputs."),
    ],
    "betavae": [
        ("beta", "Multiplier applied to the KL term in Beta-VAE."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
    ],
    "hvae": [
        ("kl_weight", "Weight applied to hierarchical KL regularization."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
        ("top_latent_dim", "Width of the top-level latent hierarchy."),
    ],
    "infovae": [
        ("kl_weight", "Weight applied to the KL portion of the objective."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
        ("mmd_weight", "Weight of the MMD prior-matching penalty."),
        ("mmd_bandwidths", "Kernel bandwidths used by the MMD estimator."),
    ],
    "mmdvae": [
        ("mmd_weight", "Weight of the MMD prior-matching penalty."),
        ("mmd_bandwidths", "Kernel bandwidths used by the MMD estimator."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
    ],
    "vamppriorvae": [
        ("kl_weight", "Weight applied to KL regularization against the VampPrior."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
        ("num_pseudo_inputs", "Number of learned pseudo-inputs that define the prior mixture."),
        ("pseudo_input_std", "Initialization scale for pseudo-input parameters."),
    ],
    "factorvae": [
        ("kl_weight", "Weight applied to the base KL regularization term."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
        ("tc_weight", "Strength of the total-correlation disentanglement penalty."),
        ("discriminator_hidden_dims", "Hidden layer sizes of the total-correlation discriminator."),
        ("discriminator_learning_rate", "Optional optimizer rate for the TC discriminator."),
        ("discriminator_steps", "How many TC discriminator updates run per training batch."),
    ],
    "dipvae": [
        ("kl_weight", "Weight applied to the base KL regularization term."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
        ("dip_weight", "Overall strength of covariance regularization."),
        ("dip_offdiag_weight", "Penalty on off-diagonal latent covariance terms."),
        ("dip_diag_weight", "Penalty on diagonal covariance drift from unit variance."),
    ],
    "betatcvae": [
        ("mutual_information_weight", "Weight on the latent mutual-information term."),
        ("total_correlation_weight", "Weight on the total-correlation disentanglement term."),
        ("dimension_wise_kl_weight", "Weight on the dimension-wise prior matching term."),
        ("free_bits", "Minimum KL floor enforced per latent dimension."),
        ("kl_warmup_epochs", "Epochs used to ramp KL from the start weight to full strength."),
        ("kl_start_weight", "Initial KL weight at the start of warmup."),
    ],
    "vqvae": [
        ("codebook_size", "Number of discrete codes available in the codebook."),
        ("commitment_weight", "Strength that keeps encoder outputs close to chosen codes."),
        ("codebook_weight", "Strength that moves code vectors toward encoder outputs."),
        ("use_ema_codebook", "Whether code vectors update by EMA instead of gradient loss."),
        ("ema_decay", "EMA decay factor for codebook updates."),
        ("ema_epsilon", "Numerical stability constant for EMA updates."),
        ("dead_code_reset", "Whether unused codes are reset after each training epoch."),
        ("dead_code_threshold", "Usage count threshold below which a code is considered dead."),
    ],
    "gumbelvq": [
        ("codebook_size", "Number of discrete codes available in the codebook."),
        ("temperature", "Softmax temperature for relaxed code assignments."),
        ("straight_through", "Whether assignments are discretized with a straight-through estimator."),
        ("commitment_weight", "Strength that keeps encoder outputs close to selected code mixtures."),
        ("dead_code_reset", "Whether unused codes are reset after each training epoch."),
        ("dead_code_threshold", "Usage count threshold below which a code is considered dead."),
    ],
    "fsq": [
        ("num_levels", "Number of scalar quantization levels per latent dimension."),
        ("commitment_weight", "Strength that keeps encoder outputs near finite scalar bins."),
    ],
    "rfsq": [
        ("num_levels", "Number of scalar quantization levels per residual stage."),
        ("num_quantizers", "How many residual scalar quantizers are stacked in sequence."),
        ("commitment_weight", "Strength that keeps encoder outputs near residual scalar bins."),
    ],
    "pqvae": [
        ("codebook_size", "Number of discrete codes in each product codebook."),
        ("num_codebooks", "How many product codebooks split the latent vector."),
        ("commitment_weight", "Strength that keeps encoder outputs close to chosen codes."),
        ("codebook_weight", "Strength that moves code vectors toward encoder outputs."),
        ("use_ema_codebook", "Whether code vectors update by EMA instead of gradient loss."),
        ("ema_decay", "EMA decay factor for codebook updates."),
        ("ema_epsilon", "Numerical stability constant for EMA updates."),
        ("dead_code_reset", "Whether unused codes are reset after each training epoch."),
        ("dead_code_threshold", "Usage count threshold below which a code is considered dead."),
    ],
    "rqvae": [
        ("codebook_size", "Number of discrete codes in each residual quantizer."),
        ("num_quantizers", "How many residual quantizers are stacked in sequence."),
        ("commitment_weight", "Strength that keeps encoder outputs close to chosen codes."),
        ("codebook_weight", "Strength that moves code vectors toward encoder outputs."),
        ("use_ema_codebook", "Whether code vectors update by EMA instead of gradient loss."),
        ("ema_decay", "EMA decay factor for codebook updates."),
        ("ema_epsilon", "Numerical stability constant for EMA updates."),
        ("dead_code_reset", "Whether unused codes are reset after each training epoch."),
        ("dead_code_threshold", "Usage count threshold below which a code is considered dead."),
    ],
    "vqvae2": [
        ("codebook_size", "Number of discrete codes shared by both hierarchy levels."),
        ("top_latent_dim", "Latent width used by the top quantization hierarchy."),
        ("commitment_weight", "Strength that keeps both hierarchy levels close to selected codes."),
        ("codebook_weight", "Strength that moves hierarchy code vectors toward encoder outputs."),
        ("use_ema_codebook", "Whether hierarchy codebooks update by EMA instead of gradient loss."),
        ("ema_decay", "EMA decay factor for codebook updates."),
        ("dead_code_reset", "Whether unused hierarchy codes are reset after each training epoch."),
        ("dead_code_threshold", "Usage count threshold below which a code is considered dead."),
    ],
}


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="glove", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument("--output-dir", default="artifacts/train-autoencoder", help="Model output directory.")
    parser.add_argument("--dim", type=int, default=50, help="Dataset embedding dimension when supported.")
    parser.add_argument("--max-vectors", type=int, default=None, help="Optional dataset cap for faster experiments.")
    parser.add_argument(
        "--encoder",
        default=None,
        help="Optional encoder name for encoder-backed datasets such as SNLI and MultiNLI.",
    )
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=128,
        help="Batch size for encoder-backed dataset materialization.",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="laion2b_s34b_b79k",
        help="CLIP pretrained checkpoint name for CLIP-backed datasets.",
    )
    parser.add_argument(
        "--clip-device",
        default=None,
        help="Optional CLIP preprocessing device override, for example `cpu` or `cuda`.",
    )
    parser.add_argument(
        "--clip-modality",
        choices=["image", "text", "both"],
        default="both",
        help="Which CLIP modality to materialize for CLIP-backed datasets.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting and training.")


def add_training_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, mps.")
    parser.add_argument(
        "--show-only-best-epochs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to persist only best-epoch summaries in the terminal output.",
    )
    parser.add_argument(
        "--advice",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to print hyperparameter tuning advice after training.",
    )


def build_dataset(args: argparse.Namespace):
    dim = args.dim
    if args.dataset in {"fasttext", "numberbatch"} and dim == 50:
        dim = 300
    if args.dataset in {"snli", "multinli"}:
        return load_dataset(
            args.dataset,
            encoder_name=args.encoder,
            encoder_batch_size=args.encoder_batch_size,
            max_vectors=args.max_vectors,
        )
    if args.dataset == "flickr30k":
        return load_dataset(
            args.dataset,
            encoder_name=args.encoder,
            encoder_pretrained=args.clip_pretrained,
            encoder_batch_size=args.encoder_batch_size,
            encoder_device=args.clip_device,
            modality=args.clip_modality,
            max_vectors=args.max_vectors,
        )
    return load_dataset(args.dataset, dim=dim, max_vectors=args.max_vectors)


def prepare_training(args: argparse.Namespace):
    set_seed(args.seed)
    dataset = build_dataset(args)
    dataloaders = dataset.get_dataloaders(
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    embedding_matrix = dataset.load_embedding_matrix()
    return dataset, dataloaders, embedding_matrix.embedding_dim


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        show_only_best_epochs=args.show_only_best_epochs,
        advice=args.advice,
    )


def _resolve_parameter_value(model, args: argparse.Namespace, name: str):
    if hasattr(model.config, name):
        return getattr(model.config, name)
    return getattr(args, name)


def _format_parameter_value(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _print_section(title: str) -> None:
    print(style(title, fg="yellow", bold=True))


def _print_parameter_row(name: str, value: str, description: str) -> None:
    label = style(f"{name:<24}", fg="cyan")
    rendered_value = style(value, fg="white", bold=True)
    rendered_description = style(description, fg="magenta", dim=True)
    print(f"  {label} {rendered_value}  {rendered_description}")


def print_training_overview(args: argparse.Namespace, model, *, input_dim: int) -> None:
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    header = style(" TRAIN PLAN ", fg="white", bg="blue", bold=True)
    summary = style(f"{args.model} on {args.dataset}", fg="white", bold=True)

    print()
    print(f"{header} {summary}")
    _print_section("Command")
    print(f"  {style(command, fg='white', bold=True)}")

    _print_section("Data")
    _print_parameter_row("dataset", args.dataset, "Dataset or cached embedding source for this run.")
    _print_parameter_row("input_dim", str(input_dim), "Embedding width seen by the autoencoder.")
    _print_parameter_row("max_vectors", _format_parameter_value(args.max_vectors), "Optional cap on the number of embedding rows used.")
    if args.dataset in {"snli", "multinli"}:
        _print_parameter_row("encoder", _format_parameter_value(args.encoder), "Sentence encoder used to materialize embeddings.")
        _print_parameter_row("encoder_batch_size", str(args.encoder_batch_size), "Batch size used while encoding raw text into embeddings.")
    if args.dataset == "flickr30k":
        _print_parameter_row("encoder", _format_parameter_value(args.encoder), "CLIP vision-text encoder used to materialize embeddings.")
        _print_parameter_row("clip_pretrained", _format_parameter_value(args.clip_pretrained), "Pretrained CLIP checkpoint identifier.")
        _print_parameter_row("clip_modality", _format_parameter_value(args.clip_modality), "Which CLIP modality embeddings are cached.")
        _print_parameter_row("encoder_batch_size", str(args.encoder_batch_size), "Batch size used while encoding CLIP features.")

    _print_section("Training")
    _print_parameter_row("output_dir", args.output_dir, "Directory where checkpoints and metrics will be written.")
    _print_parameter_row("epochs", str(args.epochs), "Maximum number of training epochs.")
    _print_parameter_row("patience", _format_parameter_value(args.patience), "Early-stop patience in epochs without validation improvement.")
    _print_parameter_row("batch_size", str(args.batch_size), "Mini-batch size for training and evaluation.")
    _print_parameter_row("learning_rate", str(args.learning_rate), "Adam optimizer learning rate.")
    _print_parameter_row("device", args.device, "Target runtime device for training.")
    _print_parameter_row("advice", _format_parameter_value(args.advice), "Whether end-of-run hyperparameter suggestions are printed.")

    _print_section("Model")
    for parameter_name, description in COMMON_MODEL_PARAMETERS:
        value = _format_parameter_value(_resolve_parameter_value(model, args, parameter_name))
        _print_parameter_row(parameter_name, value, description)

    for parameter_name, description in MODEL_PARAMETER_SECTIONS.get(args.model, []):
        value = _format_parameter_value(_resolve_parameter_value(model, args, parameter_name))
        _print_parameter_row(parameter_name, value, description)
    print()
