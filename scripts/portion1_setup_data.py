from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from mjfm_emm1.config import PortionConfig
from mjfm_emm1.data import MockTokenizer, EMM1Dataset, build_loader, maybe_load_emm1_datasets
from mjfm_emm1.utils import save_json, set_seed, throughput


REQUIREMENTS = [
    'torch==2.2.2', 'torchvision==0.17.2', 'torchaudio==2.2.2',
    'transformers==4.40.2', 'accelerate==0.28.0', 'timm==0.9.16',
    'flash-attn==2.5.8', 'einops==0.7.0', 'xformers==0.0.26',
    'datasets==2.18.0', 'webdataset==0.2.86', 'ftfy==6.1.3',
    'pyarrow==15.0.2', 'pandas==2.2.1', 'librosa==0.10.2',
    'soundfile==0.12.1', 'wandb==0.16.6', 'open-clip-torch==2.24.0',
    'pycocotools==2.0.7', 'huggingface_hub==0.22.2',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='P1 Setup + E-MM1 data pipeline')
    parser.add_argument('--work-dir', default='/kaggle/working')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--allow-mock-data', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PortionConfig(work_dir=Path(args.work_dir), cache_dir=Path(args.work_dir) / 'emm1_cache', batch_size=args.batch_size, num_workers=args.num_workers, allow_mock_data=args.allow_mock_data)
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    bundle = maybe_load_emm1_datasets(str(cfg.cache_dir), allow_mock_data=cfg.allow_mock_data)
    loader = build_loader(
        EMM1Dataset(bundle.hundred_m, tokenizer=MockTokenizer(), image_dir=str(cfg.image_dir), audio_dir=str(cfg.audio_dir), modalities=['image', 'text'], image_size=cfg.image_size),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    start = time.time()
    sample_count = 0
    preview = None
    for step, batch in enumerate(loader):
        preview = {key: tuple(value.shape) for key, value in batch.items()}
        sample_count += batch['image'].shape[0]
        if step >= 4:
            break

    summary = {
        'requirements': REQUIREMENTS,
        'rows_100m': len(bundle.hundred_m),
        'rows_1m_audio': len(bundle.audio_1m),
        'preview_batch_shapes': preview,
        'samples_per_second_estimate': throughput(sample_count, start),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only',
    }
    save_json(cfg.work_dir / 'p1_summary.json', summary)
    (cfg.work_dir / 'p1_done.flag').write_text('ok', encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
