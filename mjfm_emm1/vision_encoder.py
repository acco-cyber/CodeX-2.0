from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_qkvpacked_func
except Exception:  # pragma: no cover
    flash_attn_qkvpacked_func = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HyperAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.heads, self.head_dim)
        if flash_attn_qkvpacked_func is not None and x.is_cuda:
            attn = flash_attn_qkvpacked_func(qkv.to(torch.bfloat16)).to(x.dtype)
        else:
            q, k, v = qkv.unbind(dim=2)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        return self.proj(attn.reshape(bsz, n_tokens, dim))


class HyperBlock(nn.Module):
    def __init__(self, dim: int, heads: int, rank: int = 32):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = HyperAttention(dim, heads)
        self.ffn = SwiGLU(dim, int(dim * 8 / 3))
        self.hyper_a = nn.Linear(dim, rank, bias=False)
        self.hyper_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.hyper_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.norm1(x))
        return h + self.ffn(self.norm2(h)) + 0.1 * self.hyper_b(self.hyper_a(x))


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, dim: int = 384):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), 'b d h w -> b (h w) d')


class HybridViT(nn.Module):
    def __init__(self, dim: int = 384, heads: int = 6, layers: int = 12, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, dim=dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList([HyperBlock(dim, heads) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        if mask is not None:
            kept = []
            for sample, sample_mask in zip(x, mask):
                kept.append(sample[sample_mask])
            x = nn.utils.rnn.pad_sequence(kept, batch_first=True)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
