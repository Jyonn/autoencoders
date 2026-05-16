#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_glove.sh" train_ae.py aae \
  --output-dir artifacts/glove/aae \
  --dataset.dim 50 \
  --dataset.max_vectors 50000 \
  --model.latent_dim 16 \
  --encoder.hidden_dims "[128, 64]" \
  --encoder.activation relu \
  --model.reconstruction_loss mse \
  --model.adversarial_weight 1.0 \
  --model.discriminator_hidden_dims "[128, 64]" \
  --trainer.discriminator_steps 1 \
  "$@"
