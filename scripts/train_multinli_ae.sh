#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_ae.py" \
  --dataset multinli \
  --model ae \
  --advice \
  --output-dir artifacts/multinli/ae \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --encoder-batch-size 128 \
  --max-vectors 50000 \
  --latent-dim 64 \
  --hidden-dims 256 128 \
  --activation relu \
  --reconstruction-loss mse \
  "$@"
