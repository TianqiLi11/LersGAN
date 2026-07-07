#!/usr/bin/env bash
set -euo pipefail

ROOT="${PWD}"
ARCHIVES="${ROOT}/data/raw/source_archives"
EXTRACTED="${ROOT}/data/raw/source_extracted"
mkdir -p "$EXTRACTED"

extract_zip() {
  local archive="$1"
  local dest="$2"
  if [[ ! -s "$archive" ]]; then
    return 0
  fi
  if [[ -f "$dest/.extracted.ok" ]]; then
    return 0
  fi
  mkdir -p "$dest"
  unzip -q -n "$archive" -d "$dest"
  touch "$dest/.extracted.ok"
}

extract_rar() {
  local archive="$1"
  local dest="$2"
  if [[ ! -s "$archive" ]]; then
    return 0
  fi
  if [[ -f "$dest/.extracted.ok" ]]; then
    return 0
  fi
  mkdir -p "$dest"
  if command -v unrar >/dev/null 2>&1; then
    unrar x -o+ "$archive" "$dest/"
  elif command -v 7z >/dev/null 2>&1; then
    7z x -y "-o$dest" "$archive"
  elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$archive" -C "$dest"
  elif command -v unar >/dev/null 2>&1; then
    unar -force-overwrite -o "$dest" "$archive"
  else
    echo "No RAR extractor found for $archive. Install unrar, 7z, bsdtar, or unar." >&2
    exit 4
  fi
  touch "$dest/.extracted.ok"
}

extract_zip "$ARCHIVES/LOLdataset.zip" "$EXTRACTED/LOL_RetinexNet"
extract_zip "$ARCHIVES/SIGGRAPH17_HDR_Trainingset.zip" "$EXTRACTED/DeepHDR_Kalantari_SIGGRAPH2017/train"
extract_zip "$ARCHIVES/SIGGRAPH17_HDR_Testset.zip" "$EXTRACTED/DeepHDR_Kalantari_SIGGRAPH2017/test"
extract_rar "$ARCHIVES/SICE_Dataset_Part1.rar" "$EXTRACTED/SICE/Part1"
extract_rar "$ARCHIVES/SICE_Dataset_Part2.rar" "$EXTRACTED/SICE/Part2"
extract_zip "$ARCHIVES/VisDrone2019-DET-train.zip" "$EXTRACTED/VisDrone_DET/train"
extract_zip "$ARCHIVES/VisDrone2019-DET-val.zip" "$EXTRACTED/VisDrone_DET/val"

for archive in "$ARCHIVES"/NWPU*.zip "$ARCHIVES"/*RESISC*.zip; do
  [[ -e "$archive" ]] || continue
  extract_zip "$archive" "$EXTRACTED/NWPU_RESISC45"
done

echo "Extraction complete under $EXTRACTED"
