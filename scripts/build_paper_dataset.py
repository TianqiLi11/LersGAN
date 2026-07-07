from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the LersGAN paper dataset layout.")
    parser.add_argument("--source-config", type=Path, default=Path("configs/paper_dataset_sources.json"))
    parser.add_argument("--check-only", action="store_true", help="Audit configured local roots without writing images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON config: {path}")
    return data


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "source"


def parse_size(value: str | list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    return int(value[0]), int(value[1])


def list_images(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        return []
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        p
        for p in iterator
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part == "__MACOSX" for part in p.parts)
        and "__MACOSX" not in p.name
        and not p.name.startswith("._")
    )


def source_roots(config: dict[str, Any], group: str, key: str) -> list[dict[str, Any]]:
    return list(config["local_rebuild"][group][key])


def assert_not_disallowed(config: dict[str, Any]) -> None:
    disallowed = [str(x).lower() for x in config.get("disallowed_sources", [])]
    local_rebuild = config.get("local_rebuild", {})
    root_groups = [
        local_rebuild.get("training_candidates", {}).get("low_light_roots", []),
        local_rebuild.get("training_candidates", {}).get("normal_light_roots", []),
        local_rebuild.get("remote_sensing_test_candidates", {}).get("normal_aerial_roots", []),
    ]
    for roots in root_groups:
        for root in roots:
            text = f"{root.get('name', '')} {root.get('local_root', '')}".lower()
            for token in disallowed:
                if token in text:
                    raise ValueError(f"Disallowed source matched {token!r}: {root}")


def collect_candidates(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for root_cfg in roots:
        root = Path(root_cfg["local_root"])
        images = list_images(root, bool(root_cfg.get("recursive", True)))
        for image in images:
            candidates.append(
                {
                    "source_name": root_cfg["name"],
                    "paper_ref": root_cfg.get("paper_ref", ""),
                    "root": root,
                    "path": image,
                }
            )
    return candidates


def audit_roots(config: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "training_low_light": source_roots(config, "training_candidates", "low_light_roots"),
        "training_normal_light": source_roots(config, "training_candidates", "normal_light_roots"),
        "test_normal_aerial": source_roots(config, "remote_sensing_test_candidates", "normal_aerial_roots"),
    }
    audit: dict[str, Any] = {}
    for group_name, roots in groups.items():
        audit[group_name] = []
        for root_cfg in roots:
            root = Path(root_cfg["local_root"])
            images = list_images(root, bool(root_cfg.get("recursive", True)))
            audit[group_name].append(
                {
                    "paper_ref": root_cfg.get("paper_ref", ""),
                    "name": root_cfg["name"],
                    "local_root": str(root),
                    "exists": root.exists(),
                    "image_count": len(images),
                }
            )
    return audit


def sample_candidates(candidates: list[dict[str, Any]], count: int, seed: int, tag: str) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise RuntimeError(f"Not enough candidate images for {tag}: need {count}, found {len(candidates)}")
    rng = random.Random(f"{seed}:{tag}")
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return shuffled[:count]


def unique_output_name(item: dict[str, Any]) -> Path:
    rel = item["path"].relative_to(item["root"]) if item["path"].is_relative_to(item["root"]) else item["path"].name
    digest = hashlib.sha1(str(item["path"]).encode("utf-8")).hexdigest()[:10]
    return Path(slug(item["source_name"])) / rel.with_suffix(f".{digest}.png")


def load_rgb(path: Path, resize: tuple[int, int] | None):
    from PIL import Image, ImageOps

    image = Image.open(path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    if resize is not None and image.size != resize:
        resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        image = image.resize(resize, resample)
    return image


def save_image(image, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def scale_brightness(image, alpha: float):
    from PIL import ImageEnhance

    return ImageEnhance.Brightness(image).enhance(alpha)


def add_centered_gaussian_noise(image, sigma: float):
    from PIL import Image, ImageChops

    if sigma <= 0:
        return image
    noise = Image.effect_noise(image.size, sigma * 255.0).convert("RGB")
    return ImageChops.add(image, noise, scale=1.0, offset=-128)


def mean_luminance(image) -> float:
    from PIL import ImageStat

    return float(ImageStat.Stat(image.convert("L")).mean[0]) / 255.0


def apply_rsdark(image, condition: dict[str, Any], noise_params: dict[str, Any], rng: random.Random):

    alpha_min, alpha_max = condition["brightness_scaling_alpha"]
    alpha = rng.uniform(float(alpha_min), float(alpha_max))
    dark_image = scale_brightness(image, alpha)

    name = condition["name"]
    params = noise_params.get(name, {})
    if name == "RSDark1":
        sigma_min, sigma_max = params.get("gaussian_sigma_range", [0.01, 0.03])
        sigma = rng.uniform(float(sigma_min), float(sigma_max))
        dark_image = add_centered_gaussian_noise(dark_image, sigma)
        applied_noise = {"gaussian_sigma": sigma}
    elif name == "RSDark2":
        peak_min, peak_max = params.get("poisson_peak_range", [20, 60])
        sigma_min, sigma_max = params.get("gaussian_read_sigma_range", [0.005, 0.02])
        peak = rng.uniform(float(peak_min), float(peak_max))
        sigma = rng.uniform(float(sigma_min), float(sigma_max))
        shot_sigma = math.sqrt(max(mean_luminance(dark_image), 0.0) / max(peak, 1e-6))
        dark_image = add_centered_gaussian_noise(dark_image, shot_sigma)
        dark_image = add_centered_gaussian_noise(dark_image, sigma)
        applied_noise = {
            "poisson_peak": peak,
            "gaussian_read_sigma": sigma,
            "poisson_shot_noise": "signal_dependent_gaussian_approximation_without_numpy",
            "shot_sigma": shot_sigma,
        }
    else:
        applied_noise = {}

    return dark_image, alpha, applied_noise


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_card(output_root: Path, summary: dict[str, Any]) -> None:
    paper = summary.get("paper", {})
    paper_title = paper.get("title", "LersGAN: A GAN-Based Model for Low-Light Remote Sensing Image Enhancement")
    paper_doi = paper.get("doi", "10.1109/JSTARS.2025.3608696")
    paper_url = f"https://doi.org/{paper_doi}"
    lines = [
        "---",
        "pretty_name: LERSGAN_RS_PAPER",
        "task_categories:",
        "- image-to-image",
        "tags:",
        "- remote-sensing",
        "- low-light-image-enhancement",
        "- lersgan",
        "size_categories:",
        "- 1K<n<10K",
        "---",
        "",
        "# LERSGAN_RS_PAPER",
        "",
        "Paper-structured LersGAN dataset rebuild for low-light remote sensing image enhancement.",
        "",
        f"- Paper: [{paper_title}]({paper_url})",
        "- GitHub: https://github.com/TianqiLi11/LersGAN",
        "- Dataset: https://huggingface.co/datasets/cod-tdq/lersgan",
        "",
        "## Training Set",
        "",
        "- Setting: unsupervised, unpaired low-light and normal-light images.",
        "- Sources: paper references [11], [53], [54], [55].",
        "- Resize: 600x400.",
        "- Train: 800 low-light + 900 normal-light.",
        "- Validation: 100 low-light + 100 normal-light.",
        "",
        "## Test Set",
        "",
        "- Base normal-light aerial images: 500 images sampled from available configured candidates.",
        "- Configured paper references for aerial candidates: [56], [57]. Actual local sources are recorded in `manifests/test_manifest.csv` and `summary.json`.",
        "- RSDark1: alpha in [0.2, 0.4] plus Gaussian sensor noise.",
        "- RSDark2: alpha in [0.05, 0.2] plus Poisson-style shot noise and Gaussian read noise; this script records the exact local noise implementation in `manifests/test_manifest.csv`.",
        "",
        "## Local Rebuild Note",
        "",
        "The paper reports aggregate selected counts, not per-source selected counts. "
        "This rebuild therefore requires manually curated local candidate folders and records "
        "the actual selected files in `manifests/*.csv`.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@article{li2025lersgan,",
        "  title = {LersGAN: A GAN-Based Model for Low-Light Remote Sensing Image Enhancement},",
        "  author = {Li, TianQi},",
        "  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},",
        "  year = {2025},",
        "  doi = {10.1109/JSTARS.2025.3608696},",
        "  publisher = {IEEE}",
        "}",
        "```",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(summary["counts"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    text = "\n".join(lines)
    (output_root / "README.md").write_text(text, encoding="utf-8")
    (output_root / "DATASET_CARD.md").write_text(text, encoding="utf-8")


def build_training_split(
    *,
    output_root: Path,
    low_candidates: list[dict[str, Any]],
    normal_candidates: list[dict[str, Any]],
    counts: dict[str, Any],
    seed: int,
    resize: tuple[int, int],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()

    train_low = int(counts["train"]["low_light_unpaired"])
    val_low = int(counts["validation"]["low_light_unpaired"])
    train_normal = int(counts["train"]["normal_light_unpaired"])
    val_normal = int(counts["validation"]["normal_light_unpaired"])

    low_selected = sample_candidates(low_candidates, train_low + val_low, seed, "training_low_light")
    normal_selected = sample_candidates(normal_candidates, train_normal + val_normal, seed, "training_normal_light")

    jobs = [
        ("train", "low_light", low_selected[:train_low]),
        ("val", "low_light", low_selected[train_low:]),
        ("train", "normal_light", normal_selected[:train_normal]),
        ("val", "normal_light", normal_selected[train_normal:]),
    ]

    for split, domain, items in jobs:
        for item in items:
            image = load_rgb(item["path"], resize)
            out_path = output_root / split / domain / unique_output_name(item)
            save_image(image, out_path, overwrite)
            split_counts[f"{split}/{domain}"] += 1
            rows.append(
                {
                    "partition": "training_validation",
                    "split": split,
                    "domain": domain,
                    "paper_ref": item["paper_ref"],
                    "source_name": item["source_name"],
                    "source_path": str(item["path"]),
                    "output_path": str(out_path),
                    "width": resize[0],
                    "height": resize[1],
                }
            )

    return rows, split_counts


def build_test_sets(
    *,
    output_root: Path,
    test_candidates: list[dict[str, Any]],
    test_cfg: dict[str, Any],
    noise_params: dict[str, Any],
    seed: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    selected_count = int(test_cfg["lersgan_selected_normal_light_aerial_images"])
    selected = sample_candidates(test_candidates, selected_count, seed, "test_normal_aerial")
    conditions = list(test_cfg["generated_low_light_sets"])

    for item in selected:
        image = load_rgb(item["path"], resize=None)
        rel = unique_output_name(item)
        normal_path = output_root / "test" / "normal" / rel
        save_image(image, normal_path, overwrite)
        split_counts["test/normal"] += 1
        rows.append(
            {
                "partition": "test",
                "split": "test",
                "domain": "normal",
                "paper_ref": item["paper_ref"],
                "source_name": item["source_name"],
                "source_path": str(item["path"]),
                "output_path": str(normal_path),
                "width": image.size[0],
                "height": image.size[1],
            }
        )

        for condition in conditions:
            rng = random.Random(f"{seed}:{condition['name']}:{item['path']}")
            dark_image, alpha, applied_noise = apply_rsdark(image, condition, noise_params, rng)
            dark_path = output_root / "test" / condition["name"] / rel
            save_image(dark_image, dark_path, overwrite)
            split_counts[f"test/{condition['name']}"] += 1
            rows.append(
                {
                    "partition": "test",
                    "split": "test",
                    "domain": condition["name"],
                    "paper_ref": item["paper_ref"],
                    "source_name": item["source_name"],
                    "source_path": str(item["path"]),
                    "normal_path": str(normal_path),
                    "output_path": str(dark_path),
                    "alpha": f"{alpha:.8f}",
                    "noise": json.dumps(applied_noise, ensure_ascii=False),
                    "width": image.size[0],
                    "height": image.size[1],
                }
            )

    return rows, split_counts


def main() -> None:
    args = parse_args()
    config = load_json(args.source_config)
    assert_not_disallowed(config)

    dataset_cfg = config["dataset"]
    training_cfg = config["training_dataset"]
    test_cfg = config["remote_sensing_test_dataset"]
    local_cfg = config["local_rebuild"]
    counts_cfg = training_cfg["lersgan_aggregate_counts"]
    seed = int(dataset_cfg.get("seed", 2026))
    output_root = Path(dataset_cfg["output_root"])

    audit = audit_roots(config)
    if args.check_only:
        required = {
            "training_low_light_required": counts_cfg["train"]["low_light_unpaired"]
            + counts_cfg["validation"]["low_light_unpaired"],
            "training_normal_light_required": counts_cfg["train"]["normal_light_unpaired"]
            + counts_cfg["validation"]["normal_light_unpaired"],
            "test_normal_aerial_required": test_cfg["lersgan_selected_normal_light_aerial_images"],
        }
        print(json.dumps({"required_counts": required, "local_root_audit": audit}, indent=2, ensure_ascii=False))
        return

    low_candidates = collect_candidates(source_roots(config, "training_candidates", "low_light_roots"))
    normal_candidates = collect_candidates(source_roots(config, "training_candidates", "normal_light_roots"))
    test_candidates = collect_candidates(
        source_roots(config, "remote_sensing_test_candidates", "normal_aerial_roots")
    )

    resize = parse_size(counts_cfg["resize"])
    if resize is None:
        raise ValueError("Training resize must be configured, expected paper value 600x400.")

    training_rows, training_counts = build_training_split(
        output_root=output_root,
        low_candidates=low_candidates,
        normal_candidates=normal_candidates,
        counts=counts_cfg,
        seed=seed,
        resize=resize,
        overwrite=args.overwrite,
    )
    test_rows, test_counts = build_test_sets(
        output_root=output_root,
        test_candidates=test_candidates,
        test_cfg=test_cfg,
        noise_params=local_cfg.get("rsdark_noise_parameters_reproduction", {}),
        seed=seed,
        overwrite=args.overwrite,
    )

    manifests_dir = output_root / "manifests"
    write_manifest(manifests_dir / "train_val_manifest.csv", training_rows)
    write_manifest(manifests_dir / "test_manifest.csv", test_rows)
    write_manifest(output_root / "manifest.csv", training_rows + test_rows)

    counts = dict(training_counts + test_counts)
    summary = {
        "paper": config["paper"],
        "name": dataset_cfg["name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_config": str(args.source_config),
        "output_root": str(output_root),
        "downloaded_by_this_script": False,
        "important_note": config["important_note"],
        "counts": counts,
        "local_root_audit": audit,
        "training_dataset": training_cfg,
        "remote_sensing_test_dataset": test_cfg,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_dataset_card(output_root, summary)

    print(f"Dataset written to: {output_root}")
    print(f"Manifest written to: {output_root / 'manifest.csv'}")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
