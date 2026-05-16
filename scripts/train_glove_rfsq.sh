#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vq.py" \
  --dataset glove \
  --model rfsq \
  --output-dir artifacts/glove/rfsq \
  --dim 50 \
  --max-vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.num_levels 8 \
  --model.num_quantizers 4 \
  --model.commitment_weight 0.25 \
  --advice \
  "$@"
