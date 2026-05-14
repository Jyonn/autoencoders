#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_vq.py pqvae \
  --output-dir artifacts/glove/pqvae \
  --dim 50 \
  --max-vectors 50000 \
  --latent-dim 16 \
  --hidden-dims 128 64 \
  --activation relu \
  --reconstruction-loss mse \
  --codebook-size 256 \
  --num-codebooks 4 \
  --commitment-weight 0.25 \
  --codebook-weight 1.0 \
  --use-ema-codebook \
  --ema-decay 0.99 \
  --ema-epsilon 1e-5 \
  --dead-code-reset \
  --dead-code-threshold 0 \
  "$@"
