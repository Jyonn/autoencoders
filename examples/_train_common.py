"""Shared helpers for training entrypoints."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from autoencoders import TrainingArguments, load_dataset, set_seed
from autoencoders.data.loading import get_dataset_modules
from autoencoders.training.display import style
from _cli import _collect_declared_config_fields


DATASET_CHOICES = sorted(get_dataset_modules())

COMMON_MODEL_PARAMETERS = [
    ("latent_dim", "Latent vector width after encoding."),
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

TRAINER_PARAMETER_SECTIONS = {
    "aae": [
        ("generator_learning_rate", "Optional optimizer rate for the encoder adversarial path."),
        ("discriminator_learning_rate", "Optional optimizer rate for the discriminator."),
        ("discriminator_steps", "How many discriminator updates run per training batch."),
    ],
    "factorvae": [
        ("discriminator_learning_rate", "Optional optimizer rate for the TC discriminator."),
        ("discriminator_steps", "How many TC discriminator updates run per training batch."),
    ],
}

MODULE_PARAMETER_SECTIONS = {
    "mlp": [
        ("hidden_dims", "Hidden layer sizes used by the MLP backbone."),
        ("activation", "Non-linearity used between MLP layers."),
        ("use_bias", "Whether linear layers include trainable bias terms."),
    ],
}


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="glove", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument("--output-dir", default="artifacts/train-autoencoder", help="Model output directory.")
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


def add_backbone_args(parser: argparse.ArgumentParser, *, default_encoder: str | None = "mlp") -> None:
    parser.add_argument(
        "--encoder",
        default=default_encoder,
        help="Built-in encoder backbone name, for example `mlp`.",
    )
    parser.add_argument(
        "--decoder",
        default=None,
        help="Optional decoder backbone name. When omitted, the model may infer a reverse decoder from the encoder.",
    )


def build_dataset(args: argparse.Namespace):
    return load_dataset(args.dataset, **args.resolved_configs.dataset_config)


def prepare_training(args: argparse.Namespace):
    set_seed(args.seed)
    dataset = build_dataset(args)
    dataloaders = dataset.get_dataloaders(
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return dataset, dataloaders


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


def _print_auto_decoder_section(module_type: str | None) -> None:
    resolved_type = module_type or "backbone"
    prefix = style(f"Decoder ({resolved_type} ", fg="yellow", bold=True)
    auto = style("[auto]", fg="black", bg="yellow", bold=True)
    suffix = style(")", fg="yellow", bold=True)
    print(f"{prefix}{auto}{suffix}")


def _print_parameter_row(name: str, value: str, description: str) -> None:
    label = style(f"{name:<24}", fg="cyan")
    rendered_value = style(value, fg="white", bold=True)
    rendered_description = style(description, fg="magenta", dim=True)
    print(f"  {label} {rendered_value}  {rendered_description}")


def _print_config_section(title: str, config, descriptions: list[tuple[str, str]]) -> None:
    _print_section(title)
    description_map = dict(descriptions)
    for field_name in _collect_declared_config_fields(config.__class__):
        if not hasattr(config, field_name):
            continue
        value = _format_parameter_value(getattr(config, field_name))
        description = description_map.get(field_name, "Resolved configuration value.")
        _print_parameter_row(field_name, value, description)


def print_training_overview(args: argparse.Namespace, model, *, sample_spec) -> None:
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    header = style(" TRAIN PLAN ", fg="white", bg="blue", bold=True)
    summary = style(f"{args.model} on {args.dataset}", fg="white", bold=True)

    print()
    print(f"{header} {summary}")
    _print_section("Command")
    print(f"  {style(command, fg='white', bold=True)}")

    _print_section("Data")
    _print_parameter_row("dataset", args.dataset, "Dataset or cached embedding source for this run.")
    if hasattr(sample_spec, "shape") and sample_spec.shape:
        feature_dim = sample_spec.shape[-1]
        if feature_dim is not None:
            _print_parameter_row("input_dim", str(feature_dim), "Embedding width seen by the autoencoder.")
    _print_parameter_row("sample_spec", str(sample_spec), "Structural sample spec provided by the dataset.")
    dataset_config = args.resolved_configs.dataset_config
    if dataset_config.get("dim") is not None:
        _print_parameter_row("dim", _format_parameter_value(dataset_config["dim"]), "Dataset embedding variant selected for sources with multiple dimensions.")
    _print_parameter_row("max_vectors", _format_parameter_value(dataset_config["max_vectors"]), "Optional cap on the number of embedding rows used.")
    if args.dataset in {"snli", "multinli"}:
        _print_parameter_row("encoder", _format_parameter_value(dataset_config["encoder"]), "Sentence encoder used to materialize embeddings.")
        _print_parameter_row("encoder_batch_size", str(dataset_config["encoder_batch_size"]), "Batch size used while encoding raw text into embeddings.")
    if args.dataset == "flickr30k":
        _print_parameter_row("encoder", _format_parameter_value(dataset_config["encoder"]), "CLIP vision-text encoder used to materialize embeddings.")
        _print_parameter_row("clip_pretrained", _format_parameter_value(dataset_config["clip_pretrained"]), "Pretrained CLIP checkpoint identifier.")
        _print_parameter_row("clip_modality", _format_parameter_value(dataset_config["clip_modality"]), "Which CLIP modality embeddings are cached.")
        _print_parameter_row("encoder_batch_size", str(dataset_config["encoder_batch_size"]), "Batch size used while encoding CLIP features.")

    _print_section("Training")
    _print_parameter_row("output_dir", args.output_dir, "Directory where checkpoints and metrics will be written.")
    _print_parameter_row("epochs", str(args.epochs), "Maximum number of training epochs.")
    _print_parameter_row("patience", _format_parameter_value(args.patience), "Early-stop patience in epochs without validation improvement.")
    _print_parameter_row("batch_size", str(args.batch_size), "Mini-batch size for training and evaluation.")
    _print_parameter_row("learning_rate", str(args.learning_rate), "Adam optimizer learning rate.")
    _print_parameter_row("device", args.device, "Target runtime device for training.")
    _print_parameter_row("advice", _format_parameter_value(args.advice), "Whether end-of-run hyperparameter suggestions are printed.")
    trainer_descriptions = TRAINER_PARAMETER_SECTIONS.get(args.model, [])
    trainer_config = args.resolved_configs.trainer_config
    for field_name, description in trainer_descriptions:
        if field_name not in trainer_config:
            continue
        _print_parameter_row(field_name, _format_parameter_value(trainer_config[field_name]), description)

    _print_config_section("Model", model.config, COMMON_MODEL_PARAMETERS + MODEL_PARAMETER_SECTIONS.get(args.model, []))

    if model._encoder_module_type is not None and model.encoder is not None and hasattr(model.encoder, "config"):
        title = f"Encoder ({model._encoder_module_type})"
        _print_config_section(title, model.encoder.config, MODULE_PARAMETER_SECTIONS.get(model._encoder_module_type, []))
    elif args.encoder is not None:
        _print_section("Encoder")
        _print_parameter_row("type", _format_parameter_value(args.encoder), "Selected encoder backbone.")

    if getattr(model, "_decoder_is_auto", False):
        _print_auto_decoder_section(model._decoder_module_type or model._encoder_module_type)
    elif model._decoder_module_type is not None and model.decoder is not None and hasattr(model.decoder, "config"):
        title = f"Decoder ({model._decoder_module_type})"
        _print_config_section(title, model.decoder.config, MODULE_PARAMETER_SECTIONS.get(model._decoder_module_type, []))
    elif args.decoder is not None:
        _print_section("Decoder")
        _print_parameter_row("type", _format_parameter_value(args.decoder), "Selected decoder backbone.")
    print()


def validate_model_input_compatibility(args: argparse.Namespace, model, dataloaders) -> None:
    first_batch = next(iter(dataloaders.train))
    actual_rank = first_batch.dim()
    required_rank = model.min_input_rank
    if actual_rank >= required_rank:
        return

    header = style(" INPUT MISMATCH ", fg="white", bg="red", bold=True)
    summary = style(f"{args.model} cannot train on {args.dataset}", fg="white", bold=True)

    print()
    print(f"{header} {summary}")
    _print_section("Why this stopped")
    print(
        "  "
        + style(
            f"{model.__class__.__name__} expects each sample to contain multiple vectors, "
            f"but `{args.dataset}` currently yields single vectors with batch shape {tuple(first_batch.shape)}.",
            fg="white",
            bold=True,
        )
    )

    _print_section("What this model expects")
    _print_parameter_row("required_rank", str(required_rank), "Minimum tensor rank accepted by this model.")
    _print_parameter_row("received_shape", str(tuple(first_batch.shape)), "First training batch produced by the selected dataset.")
    _print_parameter_row("expected_examples", "B x T x D", "Typical multi-vector layout, for example token grids or spatial embedding maps.")

    _print_section("How to fix it")
    if args.model in {"vqvae", "gumbelvq", "vqvae2"}:
        print(
            "  "
            + style(
                "Use a multi-vector dataset for this model, or switch to `pqvae`, `rqvae`, `fsq`, or `rfsq` for single-vector embedding datasets such as GloVe.",
                fg="green",
                bold=True,
            )
        )
    else:
        print(
            "  "
            + style(
                "Choose a dataset that yields higher-rank samples, or select a model family that supports single-vector embeddings.",
                fg="green",
                bold=True,
            )
        )
    print()
    raise SystemExit(2)
