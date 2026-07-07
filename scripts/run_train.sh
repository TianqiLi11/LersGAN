#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --config configs/lersgan_default.yaml
