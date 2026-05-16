#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_ae.py" \
  --dataset snli \
  --model ae \
  --advice \
  --output-dir artifacts/snli/ae \
  --dataset.max_vectors 50000 \
  --dataset.encoder sentence-transformers/all-MiniLM-L6-v2 \
  --dataset.encoder_batch_size 128 \
  --model.latent_dim 64 \
  --encoder.hidden_dims "[256, 128]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  "$@"
