from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def make_relative_index(paths: Iterable[Path], base: Path) -> dict[str, Path]:
    return {str(path.relative_to(base)): path for path in paths}


class PairedEnhancementDataset(Dataset):
    """Loads paired low-light and normal-light remote sensing image chips."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        input_domain: str,
        target_domain: str = "normal",
        image_size: int = 256,
    ) -> None:
        self.root = Path(root)
        self.input_dir = self.root / split / input_domain
        self.target_dir = self.root / split / target_domain

        input_paths = list_images(self.input_dir)
        target_index = make_relative_index(list_images(self.target_dir), self.target_dir)

        self.pairs: list[tuple[Path, Path, str]] = []
        for input_path in input_paths:
            key = str(input_path.relative_to(self.input_dir))
            target_path = target_index.get(key)
            if target_path is not None:
                self.pairs.append((input_path, target_path, key))

        if not self.pairs:
            raise RuntimeError(
                f"No paired images found for split={split!r}, "
                f"input={self.input_dir}, target={self.target_dir}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        input_path, target_path, rel_path = self.pairs[index]
        low = Image.open(input_path).convert("RGB")
        normal = Image.open(target_path).convert("RGB")
        return {
            "low": self.transform(low),
            "normal": self.transform(normal),
            "name": rel_path,
            "low_path": str(input_path),
            "normal_path": str(target_path),
        }


class EvaluationEnhancementDataset(Dataset):
    """Loads an input domain and optionally matches normal-light targets."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        input_domain: str,
        target_domain: str = "normal",
        image_size: int = 256,
    ) -> None:
        self.root = Path(root)
        self.input_dir = self.root / split / input_domain
        self.target_dir = self.root / split / target_domain

        input_paths = list_images(self.input_dir)
        target_index = make_relative_index(list_images(self.target_dir), self.target_dir)

        self.samples: list[tuple[Path, Path | None, str]] = []
        for input_path in input_paths:
            key = str(input_path.relative_to(self.input_dir))
            self.samples.append((input_path, target_index.get(key), key))

        if not self.samples:
            raise RuntimeError(f"No input images found for split={split!r}, input={self.input_dir}")

        self.has_any_targets = any(target_path is not None for _, target_path, _ in self.samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        input_path, target_path, rel_path = self.samples[index]
        low = self.transform(Image.open(input_path).convert("RGB"))

        if target_path is None:
            normal = low
            normal_path = ""
            has_target = False
        else:
            normal = self.transform(Image.open(target_path).convert("RGB"))
            normal_path = str(target_path)
            has_target = True

        return {
            "low": low,
            "normal": normal,
            "has_target": has_target,
            "name": rel_path,
            "low_path": str(input_path),
            "normal_path": normal_path,
        }


class UnpairedEnhancementDataset(Dataset):
    """Loads unpaired low-light and normal-light image pools for LersGAN training."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        low_domain: str = "low_light",
        normal_domain: str = "normal_light",
        image_size: int = 256,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.low_dir = self.root / split / low_domain
        self.normal_dir = self.root / split / normal_domain
        self.low_paths = list_images(self.low_dir)
        self.normal_paths = list_images(self.normal_dir)

        if not self.low_paths:
            raise RuntimeError(f"No low-light images found in {self.low_dir}")
        if not self.normal_paths:
            raise RuntimeError(f"No normal-light images found in {self.normal_dir}")

        crop = transforms.RandomCrop(image_size, pad_if_needed=True) if augment else transforms.CenterCrop(image_size)
        flip = [transforms.RandomHorizontalFlip()] if augment else []
        self.transform = transforms.Compose(
            [
                crop,
                *flip,
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return max(len(self.low_paths), len(self.normal_paths))

    def __getitem__(self, index: int) -> dict:
        low_path = self.low_paths[index % len(self.low_paths)]
        normal_path = self.normal_paths[index % len(self.normal_paths)]
        low = Image.open(low_path).convert("RGB")
        normal = Image.open(normal_path).convert("RGB")
        return {
            "low": self.transform(low),
            "normal": self.transform(normal),
            "name": low_path.name,
            "low_path": str(low_path),
            "normal_path": str(normal_path),
        }
