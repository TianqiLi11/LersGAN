#!/usr/bin/env bash
set -euo pipefail

bash scripts/extract_source_archives.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -s data/raw/source_archives/NWPU_RESISC45_hf_train.parquet ]]; then
  "$PYTHON_BIN" scripts/extract_nwpu_hf_parquet.py
fi

"$PYTHON_BIN" scripts/populate_paper_candidates.py \
  --extracted-root data/raw/source_extracted \
  --candidate-root data/raw/lersgan_paper \
  --manifest data/raw/lersgan_paper/candidate_manifest.csv \
  --link-mode copy \
  --overwrite

"$PYTHON_BIN" scripts/build_paper_dataset.py \
  --source-config configs/paper_dataset_sources.json \
  --check-only
