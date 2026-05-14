#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <entrypoint> <model> [extra args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ENTRYPOINT="$1"
MODEL_NAME="$2"
shift 2

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/${ENTRYPOINT}" \
  --dataset glove \
  --model "${MODEL_NAME}" \
  "$@"
