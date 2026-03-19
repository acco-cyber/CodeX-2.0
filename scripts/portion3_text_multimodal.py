from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from mjfm_emm1.config import PortionConfig
from mjfm_emm1.data import MockTokenizer, EMM1Dataset, build_loader, maybe_load_emm1_datasets
from mjfm_emm1.jepa import JEPAPredictor, build_ema, extract_target_embeds, random_jepa_mask, update_ema
from mjfm_emm1.losses import MLPProjector, contrastive_loss, manifold_reg
from mjfm_emm1.text_encoder import TextEncoder
from mjfm_emm1.utils import get_device, set_seed
from mjfm_emm1.vision_encoder import HybridViT


def parse_args():
    parser = argparse.ArgumentParser(description='P3 Text + multimodal alignment')
    parser.add_argument('--work-dir', default='/kaggle/working')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--allow-mock-data', action='store_true')
    parser.add_argument('--vision-ckpt', default='/kaggle/working/ckpt_jepa_vision.pt')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PortionConfig(work_dir=Path(args.work_dir), cache_dir=Path(args.work_dir) / 'emm1_cache', batch_size=args.batch_size, num_workers=args.num_workers, total_steps=args.steps, allow_mock_data=args.allow_mock_data)
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    bundle = maybe_load_emm1_datasets(str(cfg.cache_dir), allow_mock_data=cfg.allow_mock_data)
    loader = build_loader(EMM1Dataset(bundle.hundred_m, MockTokenizer(), str(cfg.image_dir), str(cfg.audio_dir), modalities=['image', 'text'], image_size=cfg.image_size), cfg.batch_size, cfg.num_workers)

    encoder = HybridViT(dim=cfg.model_dim, heads=cfg.num_heads, layers=2, img_size=cfg.image_size, patch_size=cfg.patch_size).to(device)
    predictor = JEPAPredictor(dim=cfg.model_dim, pred_dim=192, heads=4, layers=2).to(device)
    ema_encoder = build_ema(encoder).to(device)
    if Path(args.vision_ckpt).exists():
        checkpoint = torch.load(args.vision_ckpt, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'], strict=False)
        ema_encoder.load_state_dict(checkpoint['ema_encoder'], strict=False)
        predictor.load_state_dict(checkpoint['predictor'], strict=False)

    text_encoder = TextEncoder(model_name=cfg.text_model_name, d_out=cfg.shared_dim).to(device)
    proj_vision = MLPProjector(d_in=cfg.model_dim, d_out=cfg.shared_dim).to(device)
    proj_text = MLPProjector(d_in=cfg.shared_dim, d_out=cfg.shared_dim).to(device)
    log_temp = torch.nn.Parameter(torch.tensor(0.07, device=device).log())

    params = list(encoder.parameters()) + list(predictor.parameters()) + list(text_encoder.parameters()) + list(proj_vision.parameters()) + list(proj_text.parameters()) + [log_temp]
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=device.type == 'cuda')

    step = 0
    while step < cfg.total_steps:
        for batch in loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ctx_mask, tgt_idx = random_jepa_mask(images.shape[0], n_patches=(cfg.image_size // cfg.patch_size) ** 2, ctx_ratio=cfg.ctx_ratio, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=True):
                z_ctx = encoder(images, mask=ctx_mask)
                z_pred = predictor(z_ctx, n_target_patches=tgt_idx.shape[1])
                with torch.no_grad():
                    z_full = ema_encoder(images)
                    z_tgt = extract_target_embeds(z_full, tgt_idx)
                loss_jepa = F.smooth_l1_loss(z_pred, z_tgt)
                z_img = proj_vision(encoder(images)[:, 0, :])
                z_txt = proj_text(text_encoder(input_ids, attention_mask))
                loss_clip = contrastive_loss(z_img, z_txt, log_temp)
                loss_reg = manifold_reg(proj_vision, proj_text, lam=0.01)
                loss = loss_jepa + 0.5 * loss_clip + loss_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema(encoder, ema_encoder, 0.999)
            step += 1
            print({'step': step, 'loss_jepa': round(loss_jepa.item(), 4), 'loss_clip': round(loss_clip.item(), 4)})
            if step >= cfg.total_steps:
                break

    torch.save({'step': step, 'encoder': encoder.state_dict(), 'ema': ema_encoder.state_dict(), 'predictor': predictor.state_dict(), 'text_enc': text_encoder.state_dict(), 'proj_v': proj_vision.state_dict(), 'proj_t': proj_text.state_dict(), 'log_temp': log_temp.detach().cpu()}, cfg.work_dir / 'ckpt_multimodal.pt')
    print(f'saved {(cfg.work_dir / "ckpt_multimodal.pt")}')


if __name__ == '__main__':
    main()
