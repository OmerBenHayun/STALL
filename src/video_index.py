"""Build an enriched index CSV for raw video directories (e.g. GenVideo).

Standalone script + importable module.

Usage:
    python video_index.py --real-dir /data/GenVideo/real --fake-dir /data/GenVideo/fake --output genvideo.csv
    python video_index.py --real-dir /data/GenVideo/real --fake-dir /data/GenVideo/fake --output genvideo.csv --debug-n 5

Directory layout expected:
    <real-dir>/<model>/*.mp4   → subset="real"
    <fake-dir>/<model>/*.mp4   → subset="annotated"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def downsample_frames(num_frames: int, current_fps: float, target_fps: float = 8) -> list[int]:
    """Return frame indices for downsampling from current_fps to target_fps.

    Exact paper snippet — simple ratio-based selection.

    Raises:
        ValueError: if target_fps > current_fps.
    """
    if target_fps > current_fps:
        raise ValueError(
            f"target_fps ({target_fps}) > current_fps ({current_fps}). "
            "Cannot upsample; skip this video."
        )
    ratio = current_fps / target_fps
    indices = []
    j = 0
    while True:
        idx = round(ratio * j)
        if idx >= num_frames:
            break
        indices.append(idx)
        j += 1
    return indices


def get_video_metadata(path: str) -> Optional[dict]:
    """Return fps, duration_seconds, num_frames via ffprobe.

    Returns None values on failure (caller should skip the video).
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,avg_frame_rate",
        "-print_format", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]

        # avg_frame_rate is a fraction string like "30000/1001"
        rate_str = stream.get("avg_frame_rate", "0/1")
        num, den = rate_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0

        duration = float(stream.get("duration", 0))
        num_frames = round(fps * duration) if fps > 0 else 0

        return {"fps": fps, "duration_seconds": duration, "num_frames": num_frames}
    except Exception as exc:
        warnings.warn(f"ffprobe failed for {path}: {exc}")
        return {"fps": None, "duration_seconds": None, "num_frames": None}


def compute_windows(
    downsample_idxs: list[int],
    target_fps: float = 8,
    seed: int = 42,
) -> dict:
    """Pick random contiguous windows at 1/2/3/4-second durations.

    For each available duration (8/16/24/32 frames at 8 fps), picks a random
    contiguous window from `downsample_idxs` using a fixed RNG seed.

    Returns:
        {"1_sec_idxs": list|None, "2_sec_idxs": list|None,
         "3_sec_idxs": list|None, "4_sec_idxs": list|None}
        Missing durations (video too short) → None.
    """
    rng = np.random.RandomState(seed)
    n_total = len(downsample_idxs)
    result = {}
    for dur_sec in (1, 2, 3, 4):
        n_frames = int(round(target_fps * dur_sec))
        key = f"{dur_sec}_sec_idxs"
        if n_total < n_frames:
            result[key] = None
        else:
            start = rng.randint(0, n_total - n_frames + 1)
            window = downsample_idxs[start : start + n_frames]
            result[key] = list(window)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CSV builder
# ─────────────────────────────────────────────────────────────────────────────

def _collect_paths_from_dirs(real_dir: Optional[Path], fake_dir: Optional[Path]) -> list[dict]:
    """Walk real_dir/<model>/*.mp4 and fake_dir/<model>/*.mp4."""
    records = []
    pairs = []
    if real_dir is not None:
        pairs.append((real_dir, "real"))
    if fake_dir is not None:
        pairs.append((fake_dir, "annotated"))
    for subset_path, subset_val in pairs:
        if not subset_path.exists():
            continue
        for model_dir in sorted(subset_path.iterdir()):
            if not model_dir.is_dir():
                continue
            for video_file in sorted(model_dir.glob("*.mp4")):
                records.append({
                    "video_path": str(video_file),
                    "subset": subset_val,
                    "source_model": model_dir.name,
                })
    return records


def _collect_paths(root_dir: Path) -> list[dict]:
    """Walk root_dir/real/<model>/*.mp4 and root_dir/fake/<model>/*.mp4."""
    return _collect_paths_from_dirs(root_dir / "real", root_dir / "fake")


def build_video_csv(
    output_csv: str,
    root_dir: str = None,
    real_dir: str = None,
    fake_dir: str = None,
    target_fps: float = 8,
    n_workers: int = 8,
    debug_n: Optional[int] = None,
) -> pd.DataFrame:
    """Scan video directories, probe with ffprobe, compute frame indices, write CSV.

    Args:
        output_csv: Destination CSV path.
        root_dir:   Root containing real/ and fake/ subdirs (legacy; use real_dir/fake_dir instead).
        real_dir:   Directory of real videos: <model>/*.mp4 subdirs.
        fake_dir:   Directory of fake videos: <model>/*.mp4 subdirs.
        target_fps: Target frame rate for downsampling (default 8).
        n_workers:  Thread pool size for ffprobe calls.
        debug_n:    If set, keep at most this many paths per (subset, source_model)
                    before probing (fast end-to-end test).

    Returns:
        DataFrame written to output_csv.
    """
    if root_dir is not None:
        root = Path(root_dir)
        if not root.exists():
            raise ValueError(f"Directory not found: {root_dir}")
        records = _collect_paths(root)
        label = str(root_dir)
    else:
        records = _collect_paths_from_dirs(
            Path(real_dir) if real_dir else None,
            Path(fake_dir) if fake_dir else None,
        )
        parts = [p for p in [real_dir, fake_dir] if p]
        label = " + ".join(parts)
    if not records:
        raise ValueError(f"No .mp4 files found under {label}")

    df_paths = pd.DataFrame(records)
    print(f"Found {len(df_paths)} video paths under {label}")

    # Apply debug_n limit before probing
    if debug_n is not None:
        df_paths = (
            df_paths
            .groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )
        print(f"debug-n={debug_n}: reduced to {len(df_paths)} videos")

    # Probe metadata in parallel
    print(f"Probing {len(df_paths)} videos with ffprobe ({n_workers} workers)…")
    paths = df_paths["video_path"].tolist()
    metadata_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_path = {pool.submit(get_video_metadata, p): p for p in paths}
        done = 0
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            metadata_map[p] = future.result()
            done += 1
            if done % 500 == 0 or done == len(paths):
                print(f"  probed {done}/{len(paths)}", flush=True)

    df_paths["fps"] = df_paths["video_path"].map(lambda p: metadata_map[p]["fps"])
    df_paths["duration_seconds"] = df_paths["video_path"].map(
        lambda p: metadata_map[p]["duration_seconds"]
    )
    df_paths["num_frames"] = df_paths["video_path"].map(
        lambda p: metadata_map[p]["num_frames"]
    )

    # Filter: skip if fps < target_fps or duration < 1.0 or metadata failed
    n_before = len(df_paths)
    mask_ok = (
        df_paths["fps"].notna()
        & (df_paths["fps"] >= target_fps)
        & df_paths["duration_seconds"].notna()
        & (df_paths["duration_seconds"] >= 1.0)
    )
    n_filtered = (~mask_ok).sum()
    if n_filtered:
        bad = df_paths[~mask_ok]
        for _, row in bad.iterrows():
            warnings.warn(
                f"Skipping {row['video_path']}: fps={row['fps']}, "
                f"duration={row['duration_seconds']}"
            )
    df_paths = df_paths[mask_ok].copy().reset_index(drop=True)
    print(f"Filtered {n_filtered} videos (fps<{target_fps} or duration<1s or probe failure). "
          f"{len(df_paths)} remaining.")

    # Compute downsampled indices and windows
    downsample_col = []
    window_cols: dict[str, list] = {f"{d}_sec_idxs": [] for d in (1, 2, 3, 4)}
    for _, row in df_paths.iterrows():
        idxs = downsample_frames(int(row["num_frames"]), row["fps"], target_fps)
        downsample_col.append(json.dumps(idxs))
        windows = compute_windows(idxs, target_fps=target_fps)
        for d in (1, 2, 3, 4):
            key = f"{d}_sec_idxs"
            w = windows[key]
            window_cols[key].append(json.dumps(w) if w is not None else None)

    df_paths["downsample_idxs"] = downsample_col
    for key, vals in window_cols.items():
        df_paths[key] = vals

    # Reorder columns
    cols = [
        "video_path", "subset", "source_model",
        "fps", "duration_seconds", "num_frames",
        "downsample_idxs",
        "1_sec_idxs", "2_sec_idxs", "3_sec_idxs", "4_sec_idxs",
    ]
    df_out = df_paths[cols]
    df_out.to_csv(output_csv, index=False)
    print(f"Wrote {len(df_out)} rows → {output_csv}")

    # Summary
    print("\nPer-source counts:")
    summary = (
        df_out.groupby(["subset", "source_model"])
        .size()
        .reset_index(name="count")
    )
    print(summary.to_string(index=False))

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build an enriched index CSV for a raw video directory. "
            "Requires ffprobe (from ffmpeg) on PATH."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--real-dir", required=True, metavar="DIR",
        help="Directory of real videos: <model>/*.mp4 subdirs",
    )
    parser.add_argument(
        "--fake-dir", required=True, metavar="DIR",
        help="Directory of fake videos: <model>/*.mp4 subdirs",
    )
    parser.add_argument(
        "--output", required=True, metavar="CSV",
        help="Output CSV path",
    )
    parser.add_argument(
        "--target-fps", type=float, default=8.0, metavar="FPS",
        help="Target frame rate for downsampling (default: 8)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="N",
        help="Number of parallel ffprobe workers (default: 8)",
    )
    parser.add_argument(
        "--debug", nargs="?", const=True, default=None, metavar="N",
        help=(
            "Fast end-to-end test for this script: keep at most N videos per "
            "(subset, source_model) before probing. Pass --debug N, e.g. --debug 5."
        ),
    )
    args = parser.parse_args()

    _debug_raw = args.debug
    debug_n = int(_debug_raw) if isinstance(_debug_raw, str) else None

    build_video_csv(
        output_csv=args.output,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        target_fps=args.target_fps,
        n_workers=args.workers,
        debug_n=debug_n,
    )


if __name__ == "__main__":
    main()
