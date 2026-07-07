from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired low-light/normal-light remote sensing data.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/LERSGAN_RS_REPRO_90_70"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def normalized_dataset_name(path: Path, raw_root: Path) -> str:
    rel_parts = path.relative_to(raw_root).parts
    return rel_parts[0] if rel_parts else "unknown"


def collect_images(raw_root: Path, limit_per_dataset: int | None) -> list[Path]:
    image_paths = list_images(raw_root)
    if limit_per_dataset is None:
        return image_paths

    grouped: dict[str, list[Path]] = {}
    for path in image_paths:
        grouped.setdefault(normalized_dataset_name(path, raw_root), []).append(path)

    limited: list[Path] = []
    for dataset_name in sorted(grouped):
        limited.extend(grouped[dataset_name][:limit_per_dataset])
    return sorted(limited)


def enhance_brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def save_pair(image: Image.Image, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=95)


def main() -> None:
    args = parse_args()
    if not args.raw_root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {args.raw_root}")

    image_paths = collect_images(args.raw_root, args.limit_per_dataset)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.raw_root}")

    random.seed(args.seed)
    random.shuffle(image_paths)

    manifest_path = args.output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "split",
                "dataset",
                "source_path",
                "normal_path",
                "dark90_path",
                "dark70_path",
                "brightness_dark90",
                "brightness_dark70",
            ],
        )
        writer.writeheader()

        for index, source_path in enumerate(tqdm(image_paths, desc="building paired data")):
            split = split_name(index, len(image_paths), args.train_ratio, args.val_ratio)
            dataset = normalized_dataset_name(source_path, args.raw_root)
            rel = source_path.relative_to(args.raw_root).with_suffix(".png")

            normal_path = args.output_root / split / "normal" / rel
            dark90_path = args.output_root / split / "dark90" / rel
            dark70_path = args.output_root / split / "dark70" / rel

            image = Image.open(source_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            image = ImageOps.fit(image, (args.image_size, args.image_size), method=Image.Resampling.BICUBIC)

            save_pair(image, normal_path, args.overwrite)
            save_pair(enhance_brightness(image, 0.90), dark90_path, args.overwrite)
            save_pair(enhance_brightness(image, 0.70), dark70_path, args.overwrite)

            writer.writerow(
                {
                    "split": split,
                    "dataset": dataset,
                    "source_path": str(source_path),
                    "normal_path": str(normal_path),
                    "dark90_path": str(dark90_path),
                    "dark70_path": str(dark70_path),
                    "brightness_dark90": 0.90,
                    "brightness_dark70": 0.70,
                }
            )

    print(f"Processed {len(image_paths)} images.")
    print(f"Dataset written to: {args.output_root}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
