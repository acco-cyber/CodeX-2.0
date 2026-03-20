from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch


def recall_at_k(similarity: torch.Tensor, topk: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    n_items = similarity.shape[0]
    targets = torch.arange(n_items, device=similarity.device).unsqueeze(1)
    metrics: Dict[str, float] = {}
    for k in topk:
        _, top_pred = similarity.topk(k, dim=1)
        metrics[f'R@{k}'] = ((top_pred == targets).any(dim=1).float().mean().item() * 100.0)
    return metrics
