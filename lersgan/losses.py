from __future__ import annotations

import torch
from torch import nn


class GANLoss(nn.Module):
    """Least-squares GAN loss used by many image-to-image enhancement baselines."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))
