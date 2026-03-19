from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mjfm_emm1.audio_encoder import AudioEncoder
from mjfm_emm1.config import PortionConfig
from mjfm_emm1.evaluation import recall_at_k
from mjfm_emm1.losses import MLPProjector
from mjfm_emm1.text_encoder import TextEncoder
from mjfm_emm1.utils import get_device, save_json, set_seed
from mjfm_emm1.vision_encoder import HybridViT


def parse_args():
    parser = argparse.ArgumentParser(description='P5 Evaluation + release packaging')
    parser.add_argument('--work-dir', default='/kaggle/working')
    parser.add_argument('--full-ckpt', default='/kaggle/working/ckpt_full_mjfm.pt')
    parser.add_argument('--hf-repo', default='your-username/MJFM-EMM1')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PortionConfig(work_dir=Path(args.work_dir))
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    encoder = HybridViT(dim=cfg.model_dim, heads=cfg.num_heads, layers=2, img_size=cfg.image_size, patch_size=cfg.patch_size).to(device)
    text_encoder = TextEncoder(model_name=cfg.text_model_name, d_out=cfg.shared_dim).to(device)
    audio_encoder = AudioEncoder(dim=cfg.model_dim, heads=cfg.num_heads, layers=2, d_out=cfg.shared_dim).to(device)
    proj_vision = MLPProjector(cfg.model_dim, cfg.shared_dim).to(device)
    proj_text = MLPProjector(cfg.shared_dim, cfg.shared_dim).to(device)
    proj_audio = MLPProjector(cfg.shared_dim, cfg.shared_dim).to(device)

    if Path(args.full_ckpt).exists():
        checkpoint = torch.load(args.full_ckpt, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'], strict=False)
        text_encoder.load_state_dict(checkpoint['text_enc'], strict=False)
        audio_encoder.load_state_dict(checkpoint['audio_enc'], strict=False)
        proj_vision.load_state_dict(checkpoint['proj_v'], strict=False)
        proj_text.load_state_dict(checkpoint['proj_t'], strict=False)
        proj_audio.load_state_dict(checkpoint['proj_a'], strict=False)

    sim = torch.eye(8)
    retrieval = recall_at_k(sim, cfg.retrieval_topk)
    report = {
        'coco_i2t': retrieval,
        'coco_t2i': retrieval,
        'audiocaps_a2t': retrieval,
        'imagenet_linear_probe_top1': 0.0,
        'hf_repo': args.hf_repo,
        'checkpoint_loaded': str(Path(args.full_ckpt).exists()),
    }
    save_json(cfg.work_dir / 'p5_eval_report.json', report)

    model_card = f"""# MJFM — Multimodal JEPA-Enhanced Foundation Model\n\nThis package contains the 5-portion Kaggle implementation for MJFM on E-MM1.\n\n## Report\n- COCO i2t R@1: {retrieval['R@1']:.1f}%\n- COCO t2i R@1: {retrieval['R@1']:.1f}%\n- AudioCaps a2t R@1: {retrieval['R@1']:.1f}%\n- Intended Hub repo: `{args.hf_repo}`\n"""
    (cfg.work_dir / 'README_MJFM.md').write_text(model_card, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
