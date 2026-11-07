#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/commission/main.yaml}"
EPISODES="${2:-4}"

qies-control evaluate --config "${CONFIG}" --episodes "${EPISODES}"
