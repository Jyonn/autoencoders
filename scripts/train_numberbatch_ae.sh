#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_ae.py" \
  --dataset numberbatch \
  --model ae \
  --advice \
  --output-dir artifacts/numberbatch/ae \
  --dataset.max_vectors 50000 \
  --model.latent_dim 32 \
  --encoder.hidden_dims "[256, 128]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  "$@"
