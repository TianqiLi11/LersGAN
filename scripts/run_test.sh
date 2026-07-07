#!/usr/bin/env bash
set -euo pipefail

python test.py \
  --config configs/lersgan_default.yaml \
  --split test
