from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate source-exact LersGAN candidate folders.")
    parser.add_argument("--extracted-root", type=Path, default=Path("data/raw/source_extracted"))
    parser.add_argument("--candidate-root", type=Path, default=Path("data/raw/lersgan_paper"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/lersgan_paper/candidate_manifest.csv"))
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-per-source", type=int, default=1200)
    parser.add_argument("--max-aerial-per-source", type=int, default=700)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def image_stats(path: Path) -> tuple[float, float, float]:
    from PIL import Image, ImageFilter, ImageStat

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((192, 192))
        gray = image.convert("L")
        brightness = ImageStat.Stat(gray).mean[0] / 255.0
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = ImageStat.Stat(edges).mean[0] / 255.0
        hist = gray.histogram()
        total = sum(hist) or 1
        clipped = (sum(hist[:3]) + sum(hist[-3:])) / total
    return brightness, sharpness, clipped


def grouped_by_parent(paths: list[Path]) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    for path in paths:
        groups[path.parent].append(path)
    return groups


def unique_dest(base: Path, source: Path) -> Path:
    parts = [p for p in source.parts[-4:] if p not in {os.sep, ""}]
    stem = "__".join(parts)
    return base / Path(stem).with_suffix(source.suffix.lower())


def place_file(source: Path, dest: Path, link_mode: str, overwrite: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            return
        dest.unlink()
    if link_mode == "symlink":
        dest.symlink_to(source.resolve())
    else:
        shutil.copy2(source, dest)


def select_by_brightness(paths: list[Path], *, role: str, limit: int, seed: int) -> list[Path]:
    scored: list[tuple[float, float, float, Path]] = []
    for path in paths:
        try:
            brightness, sharpness, clipped = image_stats(path)
        except Exception:
            continue
        scored.append((brightness, sharpness, clipped, path))

    if role == "low":
        filtered = [x for x in scored if x[0] <= 0.34 and x[2] < 0.35]
        ordered = sorted(filtered, key=lambda x: (x[0], -x[1]))
    elif role == "normal":
        filtered = [x for x in scored if 0.32 <= x[0] <= 0.82 and x[2] < 0.25]
        ordered = sorted(filtered, key=lambda x: (abs(x[0] - 0.55), -x[1]))
    else:
        filtered = [x for x in scored if 0.20 <= x[0] <= 0.88 and x[1] >= 0.01 and x[2] < 0.25]
        rng = random.Random(seed)
        rng.shuffle(filtered)
        ordered = sorted(filtered, key=lambda x: (-x[1], abs(x[0] - 0.55)))

    return [path for _, _, _, path in ordered[:limit]]


def select_grouped_exposures(paths: list[Path], *, role: str, limit: int) -> list[Path]:
    selected: list[Path] = []
    for _, group in grouped_by_parent(paths).items():
        if len(group) < 2:
            continue
        scored = []
        for path in group:
            try:
                brightness, sharpness, clipped = image_stats(path)
            except Exception:
                continue
            scored.append((brightness, sharpness, clipped, path))
        if not scored:
            continue
        if role == "low":
            low = min(scored, key=lambda x: x[0])
            if low[0] <= 0.40:
                selected.append(low[3])
        else:
            normal = min(scored, key=lambda x: abs(x[0] - 0.55))
            if 0.30 <= normal[0] <= 0.82 and normal[2] < 0.30:
                selected.append(normal[3])
        if len(selected) >= limit:
            break
    return selected[:limit]


def copy_selection(
    rows: list[dict],
    paths: list[Path],
    *,
    source_name: str,
    paper_ref: str,
    domain: str,
    dest_root: Path,
    link_mode: str,
    overwrite: bool,
) -> None:
    for path in paths:
        dest = unique_dest(dest_root, path)
        place_file(path, dest, link_mode, overwrite)
        rows.append(
            {
                "paper_ref": paper_ref,
                "source_name": source_name,
                "domain": domain,
                "source_path": str(path),
                "candidate_path": str(dest),
            }
        )


def find_source_root(extracted_root: Path, name: str) -> Path:
    return extracted_root / name


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    low_root = args.candidate_root / "train_candidates" / "low_light"
    normal_root = args.candidate_root / "train_candidates" / "normal_light"
    aerial_root = args.candidate_root / "test_candidates" / "normal_aerial"

    lol_paths = list_images(find_source_root(args.extracted_root, "LOL_RetinexNet"))
    lol_low = [p for p in lol_paths if "low" in str(p).lower()]
    lol_normal = [p for p in lol_paths if any(token in str(p).lower() for token in ["high", "normal"])]
    if not lol_low or not lol_normal:
        lol_low = select_by_brightness(lol_paths, role="low", limit=args.max_per_source, seed=args.seed)
        lol_normal = select_by_brightness(lol_paths, role="normal", limit=args.max_per_source, seed=args.seed)
    copy_selection(
        rows,
        lol_low[: args.max_per_source],
        source_name="LOL_RetinexNet",
        paper_ref="[11]",
        domain="low_light",
        dest_root=low_root / "LOL_RetinexNet",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    copy_selection(
        rows,
        lol_normal[: args.max_per_source],
        source_name="LOL_RetinexNet",
        paper_ref="[11]",
        domain="normal_light",
        dest_root=normal_root / "LOL_RetinexNet",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    deephdr_paths = list_images(find_source_root(args.extracted_root, "DeepHDR_Kalantari_SIGGRAPH2017"))
    deephdr_low = select_grouped_exposures(deephdr_paths, role="low", limit=args.max_per_source)
    deephdr_normal = select_grouped_exposures(deephdr_paths, role="normal", limit=args.max_per_source)
    copy_selection(
        rows,
        deephdr_low,
        source_name="DeepHDR_Kalantari_SIGGRAPH2017",
        paper_ref="[53]",
        domain="low_light",
        dest_root=low_root / "DeepHDR_Kalantari_SIGGRAPH2017",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    copy_selection(
        rows,
        deephdr_normal,
        source_name="DeepHDR_Kalantari_SIGGRAPH2017",
        paper_ref="[53]",
        domain="normal_light",
        dest_root=normal_root / "DeepHDR_Kalantari_SIGGRAPH2017",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    sice_paths = list_images(find_source_root(args.extracted_root, "SICE"))
    sice_low = select_grouped_exposures(sice_paths, role="low", limit=args.max_per_source)
    sice_normal = select_grouped_exposures(sice_paths, role="normal", limit=args.max_per_source)
    copy_selection(
        rows,
        sice_low,
        source_name="SICE",
        paper_ref="[54]",
        domain="low_light",
        dest_root=low_root / "SICE",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    copy_selection(
        rows,
        sice_normal,
        source_name="SICE",
        paper_ref="[54]",
        domain="normal_light",
        dest_root=normal_root / "SICE",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    raise_paths = list_images(find_source_root(args.extracted_root, "RAISE"))
    raise_normal = select_by_brightness(raise_paths, role="normal", limit=args.max_per_source, seed=args.seed)
    copy_selection(
        rows,
        raise_normal,
        source_name="RAISE_optional",
        paper_ref="[55]",
        domain="normal_light",
        dest_root=normal_root / "RAISE_optional",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    nwpu_paths = list_images(find_source_root(args.extracted_root, "NWPU_RESISC45"))
    nwpu_aerial = select_by_brightness(
        nwpu_paths, role="aerial", limit=args.max_aerial_per_source, seed=args.seed
    )
    copy_selection(
        rows,
        nwpu_aerial,
        source_name="NWPU_RESISC45",
        paper_ref="[56]",
        domain="normal_aerial",
        dest_root=aerial_root / "NWPU_RESISC45",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    visdrone_paths = [p for p in list_images(find_source_root(args.extracted_root, "VisDrone_DET")) if "images" in str(p)]
    if not visdrone_paths:
        visdrone_paths = list_images(find_source_root(args.extracted_root, "VisDrone_DET"))
    visdrone_aerial = select_by_brightness(
        visdrone_paths, role="aerial", limit=args.max_aerial_per_source, seed=args.seed + 1
    )
    copy_selection(
        rows,
        visdrone_aerial,
        source_name="VisDrone_DET",
        paper_ref="[57]",
        domain="normal_aerial",
        dest_root=aerial_root / "VisDrone_DET",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["paper_ref", "source_name", "domain", "source_path", "candidate_path"]
    with open(args.manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["domain"], row["source_name"])] += 1
    for (domain, source), count in sorted(counts.items()):
        print(f"{domain:14s} {source:38s} {count:5d}")
    print(f"Candidate manifest: {args.manifest}")


if __name__ == "__main__":
    main()
