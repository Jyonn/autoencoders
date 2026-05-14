#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vae.py" \
  --dataset glove \
  --model betatcvae \
  --output-dir artifacts/glove/betatcvae \
  --dim 50 \
  --max-vectors 50000 \
  --latent-dim 16 \
  --hidden-dims 128 64 \
  --activation relu \
  --reconstruction-loss mse \
  --free-bits 0.02 \
  --kl-warmup-epochs 20 \
  --kl-start-weight 0.0 \
  --mutual-information-weight 1.0 \
  --total-correlation-weight 6.0 \
  --dimension-wise-kl-weight 1.0 \
  --advice \
  "$@"
