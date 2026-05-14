#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_ae.py" \
  --dataset flickr30k \
  --model ae \
  --output-dir artifacts/flickr30k/ae \
  --encoder ViT-B-32 \
  --clip-pretrained laion2b_s34b_b79k \
  --clip-modality both \
  --encoder-batch-size 16 \
  --max-vectors 50000 \
  --latent-dim 64 \
  --hidden-dims 256 128 \
  --activation relu \
  --reconstruction-loss mse \
  "$@"
