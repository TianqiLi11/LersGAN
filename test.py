from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from lersgan.data import EvaluationEnhancementDataset
from lersgan.models import LersGenerator
from lersgan.utils import append_csv, denormalize, load_yaml, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LERSGAN reproduction model.")
    parser.add_argument("--config", type=Path, default=Path("configs/lersgan_default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--input-domain", default=None, help="Override test low-light domain, e.g. RSDark1.")
    parser.add_argument("--target-domain", default=None, help="Override paired target domain for metrics.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override test output directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_metric_array(tensor: torch.Tensor) -> np.ndarray:
    array = denormalize(tensor.detach().cpu()).permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint_path = args.checkpoint or Path(config["test"]["checkpoint"])
    output_root = args.output_dir or Path(config["test"]["output_dir"])
    target_domain = args.target_domain or config["test"].get("target_domain", "normal")
    device = torch.device(args.device)
    generator = LersGenerator(**config["model"]["generator"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    domains = [args.input_domain] if args.input_domain else list(config["test"].get("low_domains", []))
    if not domains:
        domains = [config["data"].get("input_domain", "dark70")]

    for domain in domains:
        output_dir = output_root / args.split / domain
        image_dir = output_dir / "enhanced"
        metric_path = output_dir / "metrics.csv"
        image_dir.mkdir(parents=True, exist_ok=True)
        if metric_path.exists():
            metric_path.unlink()

        dataset = EvaluationEnhancementDataset(
            root=config["data"]["root"],
            split=args.split,
            input_domain=domain,
            target_domain=target_domain,
            image_size=config["data"]["image_size"],
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        psnr_values: list[float] = []
        ssim_values: list[float] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"testing {args.split}/{domain}"):
                low = batch["low"].to(device)
                enhanced = generator(low)

                rel_name = batch["name"][0]
                save_path = image_dir / rel_name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(enhanced[0]).save(save_path)

                if bool(batch["has_target"][0]):
                    normal = batch["normal"].to(device)
                    enhanced_np = to_metric_array(enhanced[0])
                    normal_np = to_metric_array(normal[0])
                    psnr = float(peak_signal_noise_ratio(normal_np, enhanced_np, data_range=1.0))
                    ssim = float(structural_similarity(normal_np, enhanced_np, channel_axis=2, data_range=1.0))
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

                    append_csv(
                        metric_path,
                        {
                            "domain": domain,
                            "name": rel_name,
                            "input_path": batch["low_path"][0],
                            "output_path": str(save_path),
                            "target_path": batch["normal_path"][0],
                            "psnr": f"{psnr:.6f}",
                            "ssim": f"{ssim:.6f}",
                        },
                    )

        print(f"[{domain}] Saved enhanced images to {image_dir}")
        if psnr_values:
            print(f"[{domain}] Average PSNR: {np.mean(psnr_values):.4f}")
            print(f"[{domain}] Average SSIM: {np.mean(ssim_values):.4f}")
            print(f"[{domain}] Per-image metrics written to {metric_path}")
        else:
            print(f"[{domain}] No paired target images found under {args.split}/{target_domain}; metrics skipped.")


if __name__ == "__main__":
    main()
