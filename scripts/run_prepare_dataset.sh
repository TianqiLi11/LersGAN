#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_dataset.py \
  --raw-root data/raw \
  --output-root data/LERSGAN_RS_REPRO_90_70 \
  --image-size 256 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 2026
