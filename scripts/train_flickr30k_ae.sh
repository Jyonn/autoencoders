#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/train_ae.py" \
  --dataset flickr30k \
  --model ae \
  --advice \
  --output-dir artifacts/flickr30k/ae \
  --dataset.encoder ViT-B-32 \
  --dataset.clip_pretrained laion2b_s34b_b79k \
  --dataset.clip_modality both \
  --dataset.encoder_batch_size 16 \
  --max-vectors 50000 \
  --model.latent_dim 64 \
  --encoder.hidden_dims "[256, 128]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  "$@"
