from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def contrastive_loss(z_left: torch.Tensor, z_right: torch.Tensor, log_temp: torch.Tensor) -> torch.Tensor:
    batch_size = z_left.shape[0]
    temp = log_temp.exp().clamp(0.01, 0.5)
    logits = (z_left @ z_right.T) / temp
    labels = torch.arange(batch_size, device=z_left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def three_way_contrastive(z_img: torch.Tensor, z_txt: torch.Tensor, z_aud: torch.Tensor, log_temp: torch.Tensor) -> torch.Tensor:
    return (
        contrastive_loss(z_img, z_txt, log_temp)
        + contrastive_loss(z_img, z_aud, log_temp)
        + contrastive_loss(z_txt, z_aud, log_temp)
    ) / 3.0


def manifold_reg(*projectors: nn.Module, lam: float = 0.01) -> torch.Tensor:
    reg = 0.0
    for proj in projectors:
        for layer in proj.modules():
            if isinstance(layer, nn.Linear):
                spec = torch.linalg.matrix_norm(layer.weight, ord=2)
                reg = reg + (spec - 1.0).pow(2)
    return lam * reg
