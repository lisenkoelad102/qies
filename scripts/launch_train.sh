#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/commission/main.yaml}"
SEED="${2:-0}"
CKPT="${3:-checkpoints/qies_control.pt}"

qies-control train --config "${CONFIG}" --seed "${SEED}" --checkpoint "${CKPT}"
