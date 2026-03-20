from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from mjfm_emm1.audio_encoder import AudioEncoder
from mjfm_emm1.config import PortionConfig
from mjfm_emm1.data import MockTokenizer, EMM1Dataset, build_loader, maybe_load_emm1_datasets
from mjfm_emm1.jepa import JEPAPredictor, build_ema, extract_target_embeds, random_jepa_mask, update_ema
from mjfm_emm1.losses import MLPProjector, manifold_reg, three_way_contrastive
from mjfm_emm1.text_encoder import TextEncoder
from mjfm_emm1.utils import get_device, set_seed
from mjfm_emm1.vision_encoder import HybridViT


def parse_args():
    parser = argparse.ArgumentParser(description='P4 Audio fusion + full JEPA')
    parser.add_argument('--work-dir', default='/kaggle/working')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--allow-mock-data', action='store_true')
    parser.add_argument('--multimodal-ckpt', default='/kaggle/working/ckpt_multimodal.pt')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PortionConfig(work_dir=Path(args.work_dir), cache_dir=Path(args.work_dir) / 'emm1_cache', batch_size=args.batch_size, num_workers=args.num_workers, total_steps=args.steps, allow_mock_data=args.allow_mock_data)
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    bundle = maybe_load_emm1_datasets(str(cfg.cache_dir), allow_mock_data=cfg.allow_mock_data)
    ds_audio = bundle.audio_1m.filter(lambda x: x['annotation_str'] == 'good') if hasattr(bundle.audio_1m, 'filter') else bundle.audio_1m
    loader = build_loader(EMM1Dataset(ds_audio, MockTokenizer(), str(cfg.image_dir), str(cfg.audio_dir), modalities=['image', 'text', 'audio'], image_size=cfg.image_size), cfg.batch_size, cfg.num_workers)

    encoder = HybridViT(dim=cfg.model_dim, heads=cfg.num_heads, layers=2, img_size=cfg.image_size, patch_size=cfg.patch_size).to(device)
    predictor = JEPAPredictor(dim=cfg.model_dim, pred_dim=192, heads=4, layers=2).to(device)
    ema_encoder = build_ema(encoder).to(device)
    text_encoder = TextEncoder(model_name=cfg.text_model_name, d_out=cfg.shared_dim).to(device)
    audio_encoder = AudioEncoder(dim=cfg.model_dim, heads=cfg.num_heads, layers=2, d_out=cfg.shared_dim).to(device)
    proj_vision = MLPProjector(cfg.model_dim, cfg.shared_dim).to(device)
    proj_text = MLPProjector(cfg.shared_dim, cfg.shared_dim).to(device)
    proj_audio = MLPProjector(cfg.shared_dim, cfg.shared_dim).to(device)
    log_temp = torch.nn.Parameter(torch.tensor(0.07, device=device).log())

    if Path(args.multimodal_ckpt).exists():
        checkpoint = torch.load(args.multimodal_ckpt, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'], strict=False)
        predictor.load_state_dict(checkpoint['predictor'], strict=False)
        text_encoder.load_state_dict(checkpoint['text_enc'], strict=False)
        proj_vision.load_state_dict(checkpoint['proj_v'], strict=False)
        proj_text.load_state_dict(checkpoint['proj_t'], strict=False)

    params = list(encoder.parameters()) + list(predictor.parameters()) + list(text_encoder.parameters()) + list(audio_encoder.parameters()) + list(proj_vision.parameters()) + list(proj_text.parameters()) + list(proj_audio.parameters()) + [log_temp]
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=device.type == 'cuda')

    step = 0
    while step < cfg.total_steps:
        for batch in loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mels = batch['mel'].to(device)
            if mels.ndim == 3:
                mels = mels.unsqueeze(1)
            ctx_mask, tgt_idx = random_jepa_mask(images.shape[0], n_patches=(cfg.image_size // cfg.patch_size) ** 2, ctx_ratio=cfg.ctx_ratio, device=device)
            aud_ctx_mask, aud_tgt_idx = random_jepa_mask(images.shape[0], n_patches=256, ctx_ratio=cfg.ctx_ratio, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=True):
                z_ctx = encoder(images, mask=ctx_mask)
                z_pred_v = predictor(z_ctx, n_target_patches=tgt_idx.shape[1])
                with torch.no_grad():
                    z_full_v = ema_encoder(images)
                    z_tgt_v = extract_target_embeds(z_full_v, tgt_idx)
                loss_jepa_v = F.smooth_l1_loss(z_pred_v, z_tgt_v)
                z_aud_full = audio_encoder.patch_forward(mels)
                z_pred_a = predictor(z_ctx, n_target_patches=aud_tgt_idx.shape[1])
                z_tgt_a = extract_target_embeds(z_aud_full, aud_tgt_idx)
                loss_cross = F.smooth_l1_loss(z_pred_a, z_tgt_a)
                z_img = proj_vision(encoder(images)[:, 0, :])
                z_txt = proj_text(text_encoder(input_ids, attention_mask))
                z_aud = proj_audio(audio_encoder(mels))
                loss_three = three_way_contrastive(z_img, z_txt, z_aud, log_temp)
                loss_reg = manifold_reg(proj_vision, proj_text, proj_audio, lam=0.005)
                loss = loss_jepa_v + loss_cross + 0.4 * loss_three + loss_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema(encoder, ema_encoder, 0.9995)
            step += 1
            print({'step': step, 'vision_jepa': round(loss_jepa_v.item(), 4), 'cross_jepa': round(loss_cross.item(), 4), 'three_way': round(loss_three.item(), 4)})
            if step >= cfg.total_steps:
                break

    torch.save({'step': step, 'encoder': encoder.state_dict(), 'ema': ema_encoder.state_dict(), 'predictor': predictor.state_dict(), 'text_enc': text_encoder.state_dict(), 'audio_enc': audio_encoder.state_dict(), 'proj_v': proj_vision.state_dict(), 'proj_t': proj_text.state_dict(), 'proj_a': proj_audio.state_dict(), 'log_temp': log_temp.detach().cpu()}, cfg.work_dir / 'ckpt_full_mjfm.pt')
    print(f'saved {(cfg.work_dir / "ckpt_full_mjfm.pt")}')


if __name__ == '__main__':
    main()
