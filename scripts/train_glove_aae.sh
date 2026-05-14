#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_ae.py aae \
  --output-dir artifacts/glove/aae \
  --dim 50 \
  --max-vectors 50000 \
  --latent-dim 16 \
  --hidden-dims 128 64 \
  --activation relu \
  --reconstruction-loss mse \
  --adversarial-weight 1.0 \
  --discriminator-hidden-dims 128 64 \
  --discriminator-steps 1 \
  "$@"
