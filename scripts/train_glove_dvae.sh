#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_vae.py dvae \
  --output-dir artifacts/glove/dvae \
  --dim 50 \
  --max-vectors 50000 \
  --latent-dim 16 \
  --hidden-dims 128 64 \
  --activation relu \
  --reconstruction-loss mse \
  --kl-weight 0.1 \
  --free-bits 0.02 \
  --kl-warmup-epochs 20 \
  --kl-start-weight 0.0 \
  --noise-type gaussian \
  --noise-std 0.1 \
  --masking-ratio 0.3 \
  "$@"
