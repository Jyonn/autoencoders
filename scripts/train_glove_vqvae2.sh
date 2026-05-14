#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vq.py" \
  --dataset glove \
  --model vqvae2 \
  --output-dir artifacts/glove/vqvae2 \
  --dim 50 \
  --max-vectors 50000 \
  --latent-dim 16 \
  --top-latent-dim 8 \
  --hidden-dims 128 64 \
  --activation relu \
  --reconstruction-loss mse \
  --codebook-size 256 \
  --commitment-weight 0.25 \
  --codebook-weight 1.0 \
  --use-ema-codebook \
  --dead-code-reset \
  --dead-code-threshold 0 \
  --advice \
  "$@"
