from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from mjfm_emm1.config import PortionConfig
from mjfm_emm1.data import MockTokenizer, EMM1Dataset, build_loader, maybe_load_emm1_datasets
from mjfm_emm1.jepa import JEPAPredictor, build_ema, extract_target_embeds, random_jepa_mask, update_ema
from mjfm_emm1.utils import cosine_warmup_lr, ema_tau, get_device, set_seed
from mjfm_emm1.vision_encoder import HybridViT


def parse_args():
    parser = argparse.ArgumentParser(description='P2 Vision JEPA pre-training')
    parser.add_argument('--work-dir', default='/kaggle/working')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--allow-mock-data', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PortionConfig(work_dir=Path(args.work_dir), cache_dir=Path(args.work_dir) / 'emm1_cache', batch_size=args.batch_size, num_workers=args.num_workers, total_steps=args.steps, allow_mock_data=args.allow_mock_data)
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    bundle = maybe_load_emm1_datasets(str(cfg.cache_dir), allow_mock_data=cfg.allow_mock_data)
    loader = build_loader(EMM1Dataset(bundle.hundred_m, MockTokenizer(), str(cfg.image_dir), str(cfg.audio_dir), modalities=['image'], image_size=cfg.image_size), cfg.batch_size, cfg.num_workers)

    encoder = HybridViT(dim=cfg.model_dim, heads=cfg.num_heads, layers=2, img_size=cfg.image_size, patch_size=cfg.patch_size).to(device)
    predictor = JEPAPredictor(dim=cfg.model_dim, pred_dim=192, heads=4, layers=2).to(device)
    ema_encoder = build_ema(encoder).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: cosine_warmup_lr(s, cfg.warmup_steps, cfg.total_steps))
    scaler = GradScaler(enabled=device.type == 'cuda')

    step = 0
    encoder.train(); predictor.train()
    while step < cfg.total_steps:
        for batch in loader:
            images = batch['image'].to(device)
            ctx_mask, tgt_idx = random_jepa_mask(images.shape[0], n_patches=(cfg.image_size // cfg.patch_size) ** 2, ctx_ratio=cfg.ctx_ratio, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=True):
                z_ctx = encoder(images, mask=ctx_mask)
                z_pred = predictor(z_ctx, n_target_patches=tgt_idx.shape[1])
                with torch.no_grad():
                    z_full = ema_encoder(images)
                    z_tgt = extract_target_embeds(z_full, tgt_idx)
                loss = F.smooth_l1_loss(z_pred, z_tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tau = ema_tau(step, cfg.total_steps, cfg.tau_start, cfg.tau_end)
            update_ema(encoder, ema_encoder, tau)
            step += 1
            print({'step': step, 'loss': round(loss.item(), 4), 'tau': round(tau, 6)})
            if step >= cfg.total_steps:
                break

    ckpt = {'step': step, 'encoder': encoder.state_dict(), 'ema_encoder': ema_encoder.state_dict(), 'predictor': predictor.state_dict(), 'optimizer': optimizer.state_dict(), 'config': cfg.__dict__}
    torch.save(ckpt, cfg.work_dir / 'ckpt_jepa_vision.pt')
    print(f'saved {(cfg.work_dir / "ckpt_jepa_vision.pt")}')


if __name__ == '__main__':
    main()
