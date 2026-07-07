#!/usr/bin/env python3
"""Extract NWPU-RESISC45 images from the HuggingFace parquet mirror."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        default="data/raw/source_archives/NWPU_RESISC45_hf_train.parquet",
        help="HuggingFace parquet file downloaded from jonathan-roberts1/NWPU-RESISC45.",
    )
    parser.add_argument(
        "--out",
        default="data/raw/source_extracted/NWPU_RESISC45",
        help="Output directory for extracted images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of images to extract; 0 means all rows.",
    )
    return parser.parse_args()


def image_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, bytes):
            return data
    if isinstance(value, bytes):
        return value
    raise TypeError(f"Unsupported HuggingFace image cell: {type(value)!r}")


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet)
    out_dir = Path(args.out)
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        return
    if not parquet_path.is_file():
        raise SystemExit(f"Missing parquet file: {parquet_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path, columns=["image", "label"])
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    total = len(images) if args.limit <= 0 else min(args.limit, len(images))

    for index, (image_cell, label) in enumerate(zip(images[:total], labels[:total])):
        data = image_bytes(image_cell)
        class_dir = out_dir / f"class_{int(label):02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"nwpu_{index:05d}.jpg"
        if out_path.exists():
            continue
        with Image.open(BytesIO(data)) as image:
            image.convert("RGB").save(out_path, quality=95)

    marker.write_text(f"extracted={total}\nsource={parquet_path}\n", encoding="utf-8")
    print(f"Extracted {total} NWPU images to {out_dir}")


if __name__ == "__main__":
    main()
