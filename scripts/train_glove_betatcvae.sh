#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vae.py" \
  --dataset glove \
  --model betatcvae \
  --output-dir artifacts/glove/betatcvae \
  --dataset.dim 50 \
  --dataset.max_vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.free_bits 0.02 \
  --model.kl_warmup_epochs 20 \
  --model.kl_start_weight 0.0 \
  --model.mutual_information_weight 1.0 \
  --model.total_correlation_weight 6.0 \
  --model.dimension_wise_kl_weight 1.0 \
  --advice \
  "$@"
