#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_ae.py wae \
  --output-dir artifacts/glove/wae \
  --dim 50 \
  --max-vectors 50000 \
  --latent-dim 16 \
  --hidden-dims 128 64 \
  --activation relu \
  --reconstruction-loss mse \
  --mmd-weight 10.0 \
  --mmd-bandwidths 0.1 0.2 0.5 1.0 2.0 \
  "$@"
