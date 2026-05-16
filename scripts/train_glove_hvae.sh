#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_vae.py hvae \
  --output-dir artifacts/glove/hvae \
  --dataset.dim 50 \
  --dataset.max_vectors 50000 \
  --model.latent_dim 16 \
  --model.top_latent_dim 8 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.kl_weight 0.1 \
  --model.free_bits 0.02 \
  --model.kl_warmup_epochs 20 \
  --model.kl_start_weight 0.0 \
  "$@"
