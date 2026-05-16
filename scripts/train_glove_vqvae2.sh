#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vq.py" \
  --dataset glove \
  --model vqvae2 \
  --output-dir artifacts/glove/vqvae2 \
  --dataset.dim 50 \
  --dataset.max_vectors 50000 \
  --model.latent_dim 16 \
  --model.top_latent_dim 8 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.codebook_size 256 \
  --model.commitment_weight 0.25 \
  --model.codebook_weight 1.0 \
  --model.use_ema_codebook true \
  --model.dead_code_reset true \
  --model.dead_code_threshold 0 \
  --advice \
  "$@"
