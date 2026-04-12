# Supplementary — Video Sample Manifests

These CSV files contain the exact video subsets used in the STALL paper experiments, provided for reproducibility. Each file lists the specific clips sampled from a larger dataset for evaluation.

| File | Source dataset | Description |
|------|---------------|-------------|
| `msvd_sampled_videos.csv` | MSVD | Real-video clips sampled from MSVD used as the real-class reference |
| `pe_kinetics400_sampled_videos.csv` | Kinetics-400 | Real-video clips from Kinetics-400 used in cross-dataset experiments |
| `sora_sampled_videos.csv` | Sora | AI-generated clips from OpenAI Sora used in evaluation |
| `veo3_sampled_videos.csv` | VEO3 | AI-generated clips from Google VEO3 used in evaluation |

## CSV Format

Each file contains columns matching the STALL eval input format:

```
video_path, subset, source_model
```

- `video_path` — relative path to the video file (relative to the dataset root you download)
- `subset` — `"real"` for real videos, `"annotated"` for AI-generated
- `source_model` — generator or dataset name (e.g. `Sora`, `MSR-VTT`)

## Usage

To reproduce paper results using these manifests, first download the corresponding datasets, then run:

```bash
python src/eval.py --csv supplementary/csvs/sora_sampled_videos.csv \
                   --params precomputed/stall_params_vatex_dino_v3.npz
```

See [docs/download_genvideo.md](../../docs/download_genvideo.md) for dataset download instructions.
