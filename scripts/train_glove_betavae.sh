#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_vae.py betavae \
  --output-dir artifacts/glove/betavae \
  --dataset.dim 50 \
  --dataset.max_vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.beta 4.0 \
  --model.free_bits 0.02 \
  --model.kl_warmup_epochs 20 \
  --model.kl_start_weight 0.0 \
  "$@"
