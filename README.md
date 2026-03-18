# CodeX-2.0

## Added Kaggle Nemotron Inference Script

This repository now includes `nemotron_reasoning_kaggle_bulletproof.py`, a two-part Kaggle inference script for the NVIDIA Nemotron reasoning challenge.

### Highlights
- Uses the requested competition data path and model path.
- Keeps model loading local-only for Kaggle stability.
- Includes a `mamba_ssm` compatibility stub for import-time failures.
- Falls back to TF-IDF retrieval when the model cannot load or generate.
- Detects text and label columns automatically.
- Writes a submission file to `/kaggle/working/submission.csv`.

### Paths
- Data: `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge`
- Model: `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`
- Output: `/kaggle/working/submission.csv`

### Run
```bash
python nemotron_reasoning_kaggle_bulletproof.py
```
