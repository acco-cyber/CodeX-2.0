# CodeX-2.0

## MJFM × E-MM1 5-Portion Kaggle Implementation

This repository now includes a runnable, portion-by-portion implementation for the **Multimodal JEPA-Enhanced Foundation Model (MJFM)** on the **E-MM1** dataset.

### What was added
- A reusable Python package: `mjfm_emm1/`
- Five runnable Kaggle-oriented scripts under `scripts/`
- Checkpoint-oriented flow matching the requested P1 → P5 execution plan
- Mock-data fallback support so the scripts can still be smoke-tested outside Kaggle

### Portion scripts
1. `scripts/portion1_setup_data.py`
2. `scripts/portion2_vision_jepa.py`
3. `scripts/portion3_text_multimodal.py`
4. `scripts/portion4_audio_fusion.py`
5. `scripts/portion5_eval_release.py`

### Example local smoke-test flow
```bash
python scripts/portion1_setup_data.py --work-dir ./artifacts --allow-mock-data
python scripts/portion2_vision_jepa.py --work-dir ./artifacts --allow-mock-data --steps 2 --batch-size 2
python scripts/portion3_text_multimodal.py --work-dir ./artifacts --allow-mock-data --steps 2 --batch-size 2 --vision-ckpt ./artifacts/ckpt_jepa_vision.pt
python scripts/portion4_audio_fusion.py --work-dir ./artifacts --allow-mock-data --steps 2 --batch-size 2 --multimodal-ckpt ./artifacts/ckpt_multimodal.pt
python scripts/portion5_eval_release.py --work-dir ./artifacts --full-ckpt ./artifacts/ckpt_full_mjfm.pt
```

### Notes
- On Kaggle, disable mock mode and point the scripts at `/kaggle/working`.
- The code is structured so you can scale the compact local smoke-test defaults to the larger H100 training budgets from your full plan.
- The training scripts save stage checkpoints compatible with the next portion.
