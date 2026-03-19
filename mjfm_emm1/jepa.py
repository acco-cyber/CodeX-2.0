from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .vision_encoder import HyperBlock, RMSNorm


class JEPAPredictor(nn.Module):
    def __init__(self, dim: int = 384, pred_dim: int = 192, heads: int = 4, layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(dim, pred_dim)
        self.blocks = nn.ModuleList([HyperBlock(pred_dim, heads) for _ in range(layers)])
        self.norm = RMSNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context_embeds: torch.Tensor, n_target_patches: int) -> torch.Tensor:
        x = self.input_proj(context_embeds)
        masks = self.mask_token.expand(x.shape[0], n_target_patches, -1)
        x = torch.cat([x, masks], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x[:, -n_target_patches:, :])


def build_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(online: nn.Module, ema: nn.Module, tau: float) -> None:
    for p_online, p_ema in zip(online.parameters(), ema.parameters()):
        p_ema.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


def random_jepa_mask(batch_size: int, n_patches: int = 196, ctx_ratio: float = 0.25, device: str | torch.device = 'cpu'):
    n_ctx = max(1, int(n_patches * ctx_ratio))
    noise = torch.rand(batch_size, n_patches, device=device)
    shuffled = torch.argsort(noise, dim=1)
    ctx_idx = shuffled[:, :n_ctx]
    tgt_idx = shuffled[:, n_ctx:]
    ctx_mask = torch.zeros(batch_size, n_patches + 1, dtype=torch.bool, device=device)
    ctx_mask[:, 0] = True
    ctx_mask.scatter_(1, ctx_idx + 1, True)
    return ctx_mask, tgt_idx


def extract_target_embeds(full_embeds: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
    dim = full_embeds.shape[-1]
    gather_idx = tgt_idx.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(full_embeds[:, 1:, :], 1, gather_idx)
