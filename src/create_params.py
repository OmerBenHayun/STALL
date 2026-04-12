"""Create STALL calibration params (.npz) from a set of real videos.

The output .npz is a drop-in replacement for stall_params_vatex_dino_v3.npz
and can be passed to eval.py via --params.

Usage (HuggingFace, no DINOv3 needed):
    python src/create_params.py \\
        --hf-dataset OmerXYZ/comgenvid \\
        --output precomputed/my_params.npz

Usage (local directory of real .mp4 files, requires DINOv3):
    python src/create_params.py \\
        --real-dir /path/to/real/videos/ \\
        --output precomputed/my_params.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from stall import (
    create_dinov3_transform,
    diff_normalized_embeddings,
    load_dinov3_model,
    load_video_frames,
    log_likelihood,
    whitening_transform as apply_whitening,
)
from whitening_transform import WhiteningTransform
from dataset_utils import load_hf_dataset
from video_index import downsample_frames, compute_windows


# ── Whitening helpers ─────────────────────────────────────────────────────────

def _fit_whitening(flat_mat: np.ndarray) -> WhiteningTransform:
    """Fit WhiteningTransform on a 2-D array [N, D]."""
    return WhiteningTransform(data=flat_mat)


def _get_mu_W(wt: WhiteningTransform):
    """Return (mu, W) as numpy float32 arrays."""
    return wt.mean_.cpu().numpy(), wt.whitening_matrix_.cpu().numpy()


def _one_frame_per_video(embs: np.ndarray, seed: int = 42) -> np.ndarray:
    """Pick 1 random frame per video. [N, T, D] -> [N, D]."""
    rng = np.random.RandomState(seed)
    N, T, _ = embs.shape
    idxs = rng.randint(0, T, size=N)
    return embs[np.arange(N), idxs]


# ── Core calibration ──────────────────────────────────────────────────────────

def build_params(embs: np.ndarray) -> dict:
    """Compute STALL params from calibration embeddings [N, T, D].

    Spatial branch:
      - Fit WhiteningTransform on 1 random frame per video -> [N, D].
      - Compute calib_ll_spat over all frames -> [N, T].

    Temporal branch:
      - Compute L2-normalised consecutive frame diffs -> [N, T-1, D].
      - Fit WhiteningTransform on all diffs (reshaped to [(T-1)*N, D]).
      - Compute calib_ll_temp over all frame-pairs -> [N, T-1].
    """
    N, T, D = embs.shape
    print(f"Calibration set: {N} videos x {T} frames x D={D}", flush=True)

    # Spatial
    print("Spatial: fitting on 1 random frame per video…", flush=True)
    single_frames = _one_frame_per_video(embs)         # [N, D]
    wt_spat = _fit_whitening(single_frames)
    mu_spat, W_spat = _get_mu_W(wt_spat)
    calib_ll_spat = log_likelihood(apply_whitening(embs, mu_spat, W_spat))  # [N, T]
    print(f"  W_spat {W_spat.shape}  calib_ll_spat {calib_ll_spat.shape}", flush=True)

    # Temporal
    print("Temporal: fitting on all diff-normalised frames…", flush=True)
    diffs = diff_normalized_embeddings(embs)           # [N, T-1, D]
    flat_diffs = diffs.reshape((T - 1) * N, D)        # [(T-1)*N, D]
    wt_temp = _fit_whitening(flat_diffs)
    mu_temp, W_temp = _get_mu_W(wt_temp)
    calib_ll_temp = log_likelihood(apply_whitening(diffs, mu_temp, W_temp))  # [N, T-1]
    print(f"  W_temp {W_temp.shape}  calib_ll_temp {calib_ll_temp.shape}", flush=True)

    return dict(
        mu_spat=mu_spat,
        W_spat=W_spat,
        calib_ll_spat=calib_ll_spat,
        mu_temp=mu_temp,
        W_temp=W_temp,
        calib_ll_temp=calib_ll_temp,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_embs_from_hf(repo_id: str, split: str, duration: int) -> np.ndarray:
    """Load real video embeddings from a HuggingFace dataset. Returns [N, T, D]."""
    samples = load_hf_dataset(repo_id, split=split, duration=duration)
    real = [s for s in samples if s["subset"] == "real"]
    print(
        f"Using {len(real)} real videos (filtered from {len(samples)} total).",
        flush=True,
    )
    if not real:
        raise ValueError(
            "No real videos found. "
            "Real videos have subset=='real'; fakes have subset=='annotated'."
        )
    return np.concatenate([s["embs"] for s in real], axis=0)  # [N, T, D]


def load_embs_from_dir(
    real_dir: str,
    duration: int,
    dino_repo: str | None,
    dino_weights: str | None,
) -> np.ndarray:
    """Extract DINOv3 embeddings from all .mp4 files under real_dir.

    Each video is downsampled to 8 fps using the same ratio-based method as
    video_index.py (downsample_frames), then a random contiguous window of
    (duration * 8) frames is selected so the result has shape [N, T, D].
    Returns [N, T, D].
    """
    target_frames = duration * 8
    video_paths = sorted(Path(real_dir).rglob("*.mp4"))
    if not video_paths:
        raise ValueError(f"No .mp4 files found under '{real_dir}'.")
    print(f"Found {len(video_paths)} videos. Extracting embeddings…", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_dinov3_model(device, repo_dir=dino_repo, weights=dino_weights)

    from PIL import Image
    import cv2

    all_embs = []
    skipped = 0
    for i, path in enumerate(video_paths):
        # Probe fps and frame count before loading pixels.
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0 or n_frames <= 0:
            print(f"  Skipping {path.name}: could not read metadata (fps={fps}, frames={n_frames}).", flush=True)
            skipped += 1
            continue

        if fps < 8:
            print(f"  Skipping {path.name}: fps={fps:.2f} < 8.", flush=True)
            skipped += 1
            continue

        ds_idxs = downsample_frames(n_frames, fps, target_fps=8)
        if len(ds_idxs) < target_frames:
            print(
                f"  Skipping {path.name}: only {len(ds_idxs)} downsampled frames "
                f"(need {target_frames} for {duration}s at 8 fps).",
                flush=True,
            )
            skipped += 1
            continue

        # Pick a random contiguous window of target_frames.
        window = compute_windows(ds_idxs, target_fps=8, seed=42)[f"{duration}_sec_idxs"]
        idxs = window
        frames = load_video_frames(str(path), frame_indices=idxs)

        with torch.no_grad():
            tensors = [
                transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in frames
            ]
            emb = model(torch.stack(tensors).to(device)).cpu().numpy()  # [T, D]
        all_embs.append(emb)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(video_paths)} done…", flush=True)

    if not all_embs:
        raise ValueError("All videos were too short. Lower --duration or provide longer videos.")
    if skipped:
        print(f"Skipped {skipped} video(s) that were too short.", flush=True)

    embs = np.stack(all_embs)  # [N, T, D]
    print(f"Embeddings shape: {embs.shape}", flush=True)
    return embs


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create STALL calibration params (.npz) from real videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hf-dataset", metavar="REPO_ID",
        help="HuggingFace dataset repo ID with pre-computed embeddings (no DINOv3 needed).",
    )
    src.add_argument(
        "--real-dir", metavar="DIR",
        help="Directory of real .mp4 files (recursively searched). Requires DINOv3.",
    )

    parser.add_argument("--output", required=True, metavar="PATH",
                        help="Output .npz path.")
    parser.add_argument("--split", default="train",
                        help="HuggingFace dataset split (default: train).")
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Second-window duration for frame selection (default: 2).")
    parser.add_argument("--dino-repo", default=None, metavar="PATH",
                        help="Path to local DINOv3 repo clone (--real-dir mode only).")
    parser.add_argument("--dino-weights", default=None, metavar="PATH",
                        help="Path to DINOv3 .pth weights file (--real-dir mode only).")

    args = parser.parse_args()

    if args.hf_dataset:
        embs = load_embs_from_hf(args.hf_dataset, split=args.split, duration=args.duration)
    else:
        embs = load_embs_from_dir(
            args.real_dir,
            duration=args.duration,
            dino_repo=args.dino_repo,
            dino_weights=args.dino_weights,
        )

    params = build_params(embs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **params)

    print(f"\nSaved: '{out_path}'")
    for k, v in params.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
