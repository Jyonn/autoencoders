#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_vq.py vqvae \
  --output-dir artifacts/glove/vqvae \
  --dataset.dim 50 \
  --dataset.max_vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.codebook_size 256 \
  --model.commitment_weight 0.25 \
  --model.codebook_weight 1.0 \
  --model.use_ema_codebook true \
  --model.ema_decay 0.99 \
  --model.ema_epsilon 1e-5 \
  --model.dead_code_reset true \
  --model.dead_code_threshold 0 \
  "$@"
