from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class PortionConfig:
    work_dir: Path = Path('/kaggle/working')
    cache_dir: Path = Path('/kaggle/working/emm1_cache')
    image_dir: Path = Path('/kaggle/working/images')
    audio_dir: Path = Path('/kaggle/working/audio')
    batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 4
    grad_accum: int = 1
    image_size: int = 224
    patch_size: int = 16
    model_dim: int = 384
    num_heads: int = 6
    num_layers: int = 12
    shared_dim: int = 512
    text_model_name: str = 'bert-base-uncased'
    audio_duration_sec: float = 4.0
    max_text_length: int = 77
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    total_steps: int = 1000
    save_every: int = 250
    ctx_ratio: float = 0.25
    tau_start: float = 0.996
    tau_end: float = 0.9999
    seed: int = 42
    device: str = 'cuda'
    allow_mock_data: bool = True
    hf_dataset_100m: str = 'encord-team/E-MM1-100M'
    hf_dataset_1m: str = 'encord-team/E-MM1-1M'
    image_data_files: Tuple[str, ...] = ('data/nn_01.parquet',)
    audio_data_file: str = 'data/audio_to_image.parquet'
    retrieval_topk: Tuple[int, ...] = field(default_factory=lambda: (1, 5, 10))

    def ensure_dirs(self) -> None:
        for path in [self.work_dir, self.cache_dir, self.image_dir, self.audio_dir]:
            path.mkdir(parents=True, exist_ok=True)
