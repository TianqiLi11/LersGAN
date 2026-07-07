#!/usr/bin/env bash
set -euo pipefail

python scripts/build_paper_dataset.py \
  --source-config configs/paper_dataset_sources.json \
  --overwrite
