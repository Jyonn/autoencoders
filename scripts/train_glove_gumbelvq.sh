#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vq.py" \
  --dataset glove \
  --model gumbelvq \
  --output-dir artifacts/glove/gumbelvq \
  --dim 50 \
  --max-vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.codebook_size 256 \
  --model.commitment_weight 0.25 \
  --model.temperature 1.0 \
  --model.straight_through true \
  --model.dead_code_reset true \
  --model.dead_code_threshold 0 \
  --advice \
  "$@"
