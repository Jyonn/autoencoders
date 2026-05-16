#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_vae.py" \
  --dataset glove \
  --model vamppriorvae \
  --output-dir artifacts/glove/vamppriorvae \
  --dim 50 \
  --max-vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.kl_weight 0.1 \
  --model.kl_warmup_epochs 20 \
  --model.kl_start_weight 0.0 \
  --model.free_bits 0.02 \
  --model.num_pseudo_inputs 128 \
  --model.pseudo_input_std 0.01 \
  "$@"
