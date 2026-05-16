#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_ae.py wae \
  --output-dir artifacts/glove/wae \
  --dim 50 \
  --max-vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.mmd_weight 10.0 \
  --model.mmd_bandwidths "[0.1, 0.2, 0.5, 1.0, 2.0]" \
  "$@"
