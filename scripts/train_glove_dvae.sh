#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_vae.py dvae \
  --output-dir artifacts/glove/dvae \
  --dim 50 \
  --max-vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.kl_weight 0.1 \
  --model.free_bits 0.02 \
  --model.kl_warmup_epochs 20 \
  --model.kl_start_weight 0.0 \
  --model.noise_type gaussian \
  --model.noise_std 0.1 \
  --model.masking_ratio 0.3 \
  "$@"
