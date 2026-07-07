from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LersGenerator(nn.Module):
    """ResNet-style enhancement generator for low-light remote sensing images."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        channels = base_channels
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        layers += [ResBlock(channels) for _ in range(num_res_blocks)]

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(channels // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator over concatenated low/enhanced pairs."""

    def __init__(self, in_channels: int = 6, base_channels: int = 64, num_layers: int = 3) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for layer_idx in range(1, num_layers):
            next_channels = min(base_channels * 2**layer_idx, 512)
            layers += [
                nn.Conv2d(channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(next_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = next_channels

        next_channels = min(channels * 2, 512)
        layers += [
            nn.Conv2d(channels, next_channels, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(next_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(next_channels, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, low: torch.Tensor, candidate: torch.Tensor | None = None) -> torch.Tensor:
        if candidate is None:
            return self.net(low)
        return self.net(torch.cat([low, candidate], dim=1))


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "Norm" in classname and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
