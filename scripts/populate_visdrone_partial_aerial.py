#!/usr/bin/env python3
"""Populate normal-aerial candidates from a recovered VisDrone partial archive."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/source_extracted/VisDrone_DET/train_partial"),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/lersgan_paper/test_candidates/normal_aerial/VisDrone_DET"),
    )
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/lersgan_paper/visdrone_partial_aerial_manifest.csv"),
    )
    return parser.parse_args()


def image_stats(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        gray = image.convert("L")
        brightness = ImageStat.Stat(gray).mean[0] / 255.0
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = ImageStat.Stat(edges).mean[0] / 255.0
        histogram = gray.histogram()
        total = max(sum(histogram), 1)
        clipped = (sum(histogram[:3]) + sum(histogram[-3:])) / total
    return brightness, sharpness, clipped


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, str | float]] = []

    for path in iter_images(args.source):
        if len(selected) >= args.limit:
            break
        try:
            brightness, sharpness, clipped = image_stats(path)
        except Exception:
            continue
        if not (0.18 <= brightness <= 0.90):
            continue
        if sharpness < 0.012 or clipped > 0.20:
            continue
        rel = path.relative_to(args.source)
        out_path = args.dest / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)
        selected.append(
            {
                "source_path": str(path),
                "output_path": str(out_path),
                "brightness": f"{brightness:.6f}",
                "sharpness": f"{sharpness:.6f}",
                "clipped": f"{clipped:.6f}",
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "output_path", "brightness", "sharpness", "clipped"])
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected {len(selected)} VisDrone partial normal-aerial candidates into {args.dest}")


if __name__ == "__main__":
    main()
