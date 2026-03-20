from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import HyperBlock, RMSNorm


class AudioPatchEmbed(nn.Module):
    def __init__(self, dim: int = 384, patch_h: int = 8, patch_w: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class AudioEncoder(nn.Module):
    def __init__(self, dim: int = 384, heads: int = 6, layers: int = 6, d_out: int = 512):
        super().__init__()
        self.patch_embed = AudioPatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, dim))
        self.blocks = nn.ModuleList([HyperBlock(dim, heads) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, d_out))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def patch_forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(mel)
        cls = self.cls_token.expand(mel.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.patch_forward(mel)
        return F.normalize(self.proj(x[:, 0, :]), dim=-1)


def mock_wav_to_mel(duration_sec: float = 4.0, sr: int = 16000, n_mels: int = 128) -> torch.Tensor:
    n_frames = 256
    rng = np.random.default_rng(42)
    mel = rng.normal(size=(n_mels, n_frames)).astype('float32')
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
    return torch.tensor(mel).unsqueeze(0)
