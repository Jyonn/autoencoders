#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vae.py" \
  --dataset glove \
  --model dipvae \
  --output-dir artifacts/glove/dipvae \
  --dim 50 \
  --max-vectors 50000 \
  --latent-dim 16 \
  --hidden-dims 128 64 \
  --activation relu \
  --reconstruction-loss mse \
  --kl-weight 0.1 \
  --free-bits 0.02 \
  --kl-warmup-epochs 20 \
  --kl-start-weight 0.0 \
  --dip-weight 10.0 \
  --dip-offdiag-weight 1.0 \
  --dip-diag-weight 1.0 \
  --advice \
  "$@"
