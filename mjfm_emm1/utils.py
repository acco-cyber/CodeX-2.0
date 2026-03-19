from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_hint: str = 'cuda') -> torch.device:
    if device_hint == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_tau(step: int, total_steps: int, tau_start: float, tau_end: float) -> float:
    cosine = (math.cos(math.pi * min(step / max(total_steps, 1), 1.0)) + 1.0) / 2.0
    return tau_end - (tau_end - tau_start) * cosine


def save_json(path: os.PathLike[str] | str, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def throughput(samples: int, start_time: float) -> float:
    elapsed = max(time.time() - start_time, 1e-6)
    return samples / elapsed
