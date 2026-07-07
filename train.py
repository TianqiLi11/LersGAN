from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lersgan.data import PairedEnhancementDataset, UnpairedEnhancementDataset
from lersgan.losses import GANLoss, l1_loss
from lersgan.models import LersGenerator, PatchDiscriminator, weights_init
from lersgan.utils import save_triplet, save_yaml, seed_everything, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LERSGAN reproduction model.")
    parser.add_argument("--config", type=Path, default=Path("configs/lersgan_default.yaml"))
    parser.add_argument("--data-root", type=Path, default=None, help="Override data.root from the config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override experiment.output_dir from the config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs from the config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size from the config.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers from the config.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.resume is not None:
        config["train"]["resume"] = str(args.resume)
    return config


def make_dataloader(config: dict, split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    if data_cfg.get("mode", "paired") == "unpaired":
        dataset = UnpairedEnhancementDataset(
            root=data_cfg["root"],
            split=split,
            low_domain=data_cfg.get("low_domain", "low_light"),
            normal_domain=data_cfg.get("normal_domain", "normal_light"),
            image_size=data_cfg["image_size"],
            augment=shuffle,
        )
    else:
        dataset = PairedEnhancementDataset(
            root=data_cfg["root"],
            split=split,
            input_domain=data_cfg["input_domain"],
            target_domain=data_cfg["target_domain"],
            image_size=data_cfg["image_size"],
        )
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=shuffle,
    )


def save_checkpoint(
    path: Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": config,
        },
        path,
    )


def maybe_resume(
    checkpoint_path: str | None,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    if not checkpoint_path:
        return 1, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint.get("global_step", 0))


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_yaml(args.config), args)
    seed_everything(config["experiment"]["seed"])

    device = torch.device(args.device)
    output_dir = Path(config["experiment"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")

    train_loader = make_dataloader(config, "train", shuffle=True)
    unpaired = config["data"].get("mode", "paired") == "unpaired"

    generator = LersGenerator(**config["model"]["generator"]).to(device)
    discriminator = PatchDiscriminator(**config["model"]["discriminator"]).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    train_cfg = config["train"]
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg["lr_g"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_cfg["lr_d"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )

    gan_loss = GANLoss().to(device)
    start_epoch, global_step = maybe_resume(
        train_cfg.get("resume"),
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        device,
    )

    log_path = log_dir / "train.log"
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        generator.train()
        discriminator.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{train_cfg['epochs']}")

        for batch in progress:
            global_step += 1
            low = batch["low"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)

            with torch.no_grad():
                enhanced_detached = generator(low)

            optimizer_d.zero_grad(set_to_none=True)
            if unpaired:
                pred_real = discriminator(normal)
                pred_fake = discriminator(enhanced_detached.detach())
            else:
                pred_real = discriminator(low, normal)
                pred_fake = discriminator(low, enhanced_detached.detach())
            loss_d = 0.5 * (gan_loss(pred_real, True) + gan_loss(pred_fake, False))
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            enhanced = generator(low)
            pred_fake_for_g = discriminator(enhanced) if unpaired else discriminator(low, enhanced)
            loss_g_gan = gan_loss(pred_fake_for_g, True)
            if unpaired:
                identity = generator(normal)
                loss_recon = l1_loss(identity, normal) * train_cfg.get("lambda_identity", 0.0)
            else:
                loss_recon = l1_loss(enhanced, normal) * train_cfg["lambda_l1"]
            loss_g = loss_g_gan + loss_recon
            loss_g.backward()
            optimizer_g.step()

            progress.set_postfix(
                loss_g=f"{loss_g.item():.4f}",
                loss_d=f"{loss_d.item():.4f}",
                recon=f"{loss_recon.item():.4f}",
            )

            if global_step % train_cfg["log_every"] == 0:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
                        f"epoch={epoch},step={global_step},"
                        f"loss_g={loss_g.item():.6f},"
                        f"loss_d={loss_d.item():.6f},"
                        f"loss_recon={loss_recon.item():.6f}\n"
                    )

            if global_step % train_cfg["sample_every"] == 0:
                save_triplet(low[:4], enhanced[:4], normal[:4], sample_dir / f"step_{global_step:08d}.png")

        save_checkpoint(
            ckpt_dir / "latest.pt",
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            epoch,
            global_step,
            config,
        )
        if epoch % train_cfg["save_every_epoch"] == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch,
                global_step,
                config,
            )

    print(f"Training finished. Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
