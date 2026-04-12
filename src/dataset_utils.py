"""Dataset loading utilities for STALL evaluation.

Four loading modes:
  load_hf_dataset        — HuggingFace Hub (pre-computed embeddings, no DINOv3 needed)
  load_local_dir         — directory convention: root/real/<model>/*.mp4, root/fake/<model>/*.mp4
  load_csv               — explicit CSV with columns: video_path, subset, source_model
  load_csv_with_emb_cache — enriched CSV from video_index.py + on-disk embedding cache
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import warnings

import numpy as np
import pandas as pd

SUBSET_TO_FOLDER = {"real": "real", "annotated": "fake"}


def _is_missing_window(val) -> bool:
    """Return True if a window-index value is absent (None or NaN from pandas)."""
    return val is None or (isinstance(val, float) and np.isnan(val))


def _get_cache_path(cache_root: Path, subset: str, source_model: str, stem: str, duration_sec: int, compact: bool) -> Path:
    """Return the .pt cache file path for a video."""
    if compact:
        return cache_root / subset / source_model / f"{stem}_{duration_sec}s.pt"
    return cache_root / subset / source_model / f"{stem}.pt"


def _slice_window(full_emb, downsample_idxs, window_idxs, video_path):
    """Slice full_emb to the window using the downsample index map.

    Returns the sliced array, or None if any window index is missing (caller should skip).
    """
    idx_to_pos = {native: pos for pos, native in enumerate(downsample_idxs)}
    positions = [idx_to_pos[i] for i in window_idxs if i in idx_to_pos]
    if len(positions) != len(window_idxs):
        warnings.warn(
            f"{video_path}: {len(window_idxs) - len(positions)} window indices "
            "not found in downsample_idxs — skipping video."
        )
        return None
    return full_emb[positions]


def load_hf_dataset(repo_id: str, split: str = "train", duration: int = 2, verbose: bool = False) -> list[dict]:
    """Load from HuggingFace Hub using pre-computed embeddings.parquet.

    No DINOv3 required — scores are computed directly from stored embeddings.

    Returns a list of dicts (all loading happens before returning):
        {"embs": np.ndarray [1, T, D], "subset": str, "source_model": str, "filename": str}

    Args:
        duration: Which second-window to use (1/2/3/4). Selects the corresponding
                  ``<duration>_sec_idxs`` column. Default: 2.
        verbose:  If True, print metadata for videos missing the requested index column.

    Dependencies: huggingface_hub, datasets, pandas, pyarrow
    """
    from huggingface_hub import hf_hub_download
    import datasets as hf_datasets

    # Download embeddings.parquet once (cached in HF cache dir)
    print("Downloading embeddings.parquet…", flush=True)
    parquet_path = hf_hub_download(
        repo_id=repo_id, filename="embeddings.parquet", repo_type="dataset"
    )
    print("Loading embeddings into memory…", flush=True)
    emb_df = pd.read_parquet(parquet_path)
    # Each "dino_embedding" cell is a numpy object array of shape (T,) where
    # each element is a 1-D float array of length D. np.stack converts it to (T, D).
    emb_lookup: dict[str, np.ndarray] = {
        row["file_name"]: np.stack(row["dino_embedding"]).astype(np.float32)
        for _, row in emb_df.iterrows()
    }

    ds = hf_datasets.load_dataset(repo_id, split=split, streaming=True)
    # Disable video decoding so the raw .mp4 files are not downloaded
    if "video" in ds.features:
        ds = ds.cast_column("video", hf_datasets.Video(decode=False))

    samples = []
    n_missing_idxs = 0
    missing_idxs_meta = []
    for sample in ds:
        filename = sample["filename"]
        folder = SUBSET_TO_FOLDER.get(sample["subset"], "fake")
        key = f"videos/{folder}/{sample['source_model']}/{filename}"
        emb = emb_lookup.get(key)
        if emb is None:
            continue

        # Select the requested duration window at 8 FPS.
        # Videos too short for the requested window (primary key is None) are skipped.
        _PRIMARY_KEY = f"{duration}_sec_idxs"
        frame_idxs = sample.get(_PRIMARY_KEY)
        if frame_idxs is not None:
            idxs = np.array(frame_idxs, dtype=int)
            emb = emb[idxs]  # (T_native, D) → (duration*8, D)
        else:
            # Video is too short for the requested duration window — skip it.
            n_missing_idxs += 1
            missing_idxs_meta.append({k: v for k, v in sample.items() if k != "video"})
            continue

        samples.append({
            "embs": emb[np.newaxis],  # [1, T, D]
            "subset": sample["subset"],
            "source_model": sample["source_model"],
            "filename": filename,
        })

    if n_missing_idxs:
        print(
            f"  Warning: {n_missing_idxs} videos missing '{duration}_sec_idxs' "
            f"(too short for the requested window, skipped). Pass --debug for details.",
            flush=True,
        )
        if verbose:
            for meta in missing_idxs_meta:
                meta_str = ", ".join(f"{k}={v!r}" for k, v in meta.items())
                print(f"    {meta_str}", flush=True)
    print(f"Loaded {len(samples)} videos. Starting scoring…", flush=True)
    return samples


def load_local_dir(root_dir: str) -> Iterator[dict]:
    """Walk root_dir/real/<model>/*.mp4 and root_dir/fake/<model>/*.mp4.

    subset is derived from the top-level folder:
        real/  → "real"
        fake/  → "annotated"
    source_model is the immediate subdirectory name.

    Yields dicts:
        {"video_path": str, "subset": str, "source_model": str}
    """
    root = Path(root_dir)
    for folder, subset_val in [("real", "real"), ("fake", "annotated")]:
        subset_path = root / folder
        if not subset_path.exists():
            continue
        for model_dir in sorted(subset_path.iterdir()):
            if not model_dir.is_dir():
                continue
            for video_file in sorted(model_dir.glob("*.mp4")):
                yield {
                    "video_path": str(video_file),
                    "subset": subset_val,
                    "source_model": model_dir.name,
                }


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV requiring columns: video_path, subset, source_model.

    Returns a pandas DataFrame directly.

    Raises:
        ValueError: if any required column is missing.
    """
    df = pd.read_csv(csv_path)
    required = {"video_path", "subset", "source_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required columns: {sorted(missing)}\n"
            f"Required: video_path, subset ('real'/'annotated'), source_model"
        )
    return df


def count_cache_misses(
    csv_path: str,
    emb_cache_dir: str,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
) -> int:
    """Count how many videos in the CSV are missing from the embedding cache.

    Fast: only checks file existence, no I/O or model calls.
    """
    df = load_csv(csv_path)
    window_col = f"{duration_sec}_sec_idxs"

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(emb_cache_dir)
    count = 0
    for _, row in df.iterrows():
        stem = Path(row["video_path"]).stem
        if compact and _is_missing_window(row.get(window_col)):
            continue
        cache_path = _get_cache_path(cache_root, row["subset"], row["source_model"], stem, duration_sec, compact)
        if not cache_path.exists():
            count += 1
    return count


def prefill_emb_cache(
    csv_path: str,
    emb_cache_dir: str,
    model,
    batch_size: int = 32,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
    num_workers: int = 4,
    video_batch: int = 8,
):
    """Phase 1: extract and cache DINOv3 embeddings for all cache-miss videos.

    Cache hits are yielded immediately. Cache misses are processed in parallel
    video-batches: ``num_workers`` threads decode videos simultaneously, then a
    single GPU pass runs over all flattened frames from the batch (cross-video
    batching for higher GPU utilization), and results are saved atomically.

    Args:
        csv_path:      Path to CSV produced by video_index.py.
        emb_cache_dir: Root directory for per-video .pt embedding files.
        model:         STALL instance used for cache-miss extraction.
        batch_size:    Frames per DINOv3 forward pass.
        duration_sec:  Used only to validate that the window column exists.
        debug_n:       If set, process at most this many rows per (subset, source_model).
        compact:       If True, extract only the --duration-second window frames instead
                       of the full downsampled video. Saves a compact cache named
                       ``{stem}_{duration_sec}s.pt``.
        num_workers:   Number of parallel CPU threads for video decoding (default: 4).
        video_batch:   Number of videos to batch together for a single GPU pass (default: 8).

    Yields:
        video_path (str) for each row processed.
    """
    import torch
    from stall import load_video_frames

    df = load_csv(csv_path)

    idx_col = "downsample_idxs"
    window_col = f"{duration_sec}_sec_idxs"
    missing_cols = [c for c in (idx_col, window_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV '{csv_path}' is missing columns: {missing_cols}\n"
            f"Re-run video_index.py to generate an enriched CSV with frame indices."
        )

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(emb_cache_dir)

    # Pass 1: classify rows as cache hits or misses without doing any I/O.
    misses = []  # (video_path, frame_idxs, cache_path) tuples to decode+embed

    for _, row in df.iterrows():
        video_path = row["video_path"]
        stem = Path(video_path).stem
        subset = row["subset"]
        source_model = row["source_model"]

        if compact:
            window_idxs_raw = row[window_col]
            if _is_missing_window(window_idxs_raw):
                continue
            frame_idxs_raw = window_idxs_raw
        else:
            frame_idxs_raw = row[idx_col]

        cache_path = _get_cache_path(cache_root, subset, source_model, stem, duration_sec, compact)
        if not cache_path.exists():
            misses.append((video_path, json.loads(frame_idxs_raw), cache_path))

    if not misses:
        return

    # Pass 2: process cache misses in parallel video-batches.
    def _decode(job):
        path, frame_idxs, _ = job
        return load_video_frames(path, frame_idxs)

    for chunk_start in range(0, len(misses), video_batch):
        chunk = misses[chunk_start : chunk_start + video_batch]

        # Decode all videos in the chunk in parallel.
        with ThreadPoolExecutor(max_workers=min(num_workers, len(chunk))) as executor:
            chunk_frames = list(executor.map(_decode, chunk))

        # Flatten all frames into one sequence, track per-video lengths.
        lengths = [len(f) for f in chunk_frames]
        flat_frames = [fr for vid in chunk_frames for fr in vid]

        # Single GPU pass over all flattened frames.
        flat_embs = model._embed_flat_frames(flat_frames, batch_size)

        # Split embeddings back per video and save atomically.
        cursor = 0
        for emb_len, (video_path, _, cache_path) in zip(lengths, chunk):
            emb = flat_embs[cursor : cursor + emb_len]
            cursor += emb_len
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.pt")
            torch.save(torch.from_numpy(emb), tmp)
            tmp.rename(cache_path)
            yield video_path


def load_csv_with_emb_cache(
    csv_path: str,
    emb_cache_dir: str,
    model,
    batch_size: int = 32,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
):
    """Load enriched CSV (from video_index.py) and yield samples with embedding cache.

    For each video row:
      - Cache miss: loads frames, extracts DINOv3 embeddings, saves to disk.
      - Cache hit: loads embeddings from disk.
    Then slices to the requested duration window.

    Args:
        csv_path:      Path to CSV produced by video_index.py.
        emb_cache_dir: Root directory for per-video .pt embedding files.
        model:         STALL instance (used for cache-miss extraction).
        batch_size:    Frames per DINOv3 forward pass on cache miss.
        duration_sec:  Which window to use: 1, 2, 3, or 4.
        debug_n:       If set, yield at most this many rows per (subset, source_model).
        compact:       If True, look for a compact cache (``{stem}_{duration_sec}s.pt``)
                       written by ``prefill_emb_cache`` with the same flag. When found,
                       the cache already contains exactly the window frames so no
                       re-indexing is needed.

    Yields:
        {"embs": np.ndarray [1, T, D], "subset": str, "source_model": str, "filename": str}

    Raises:
        ValueError: if required index columns are absent (re-run video_index.py).
    """
    import torch
    from stall import load_video_frames

    df = load_csv(csv_path)

    idx_col = "downsample_idxs"
    window_col = f"{duration_sec}_sec_idxs"
    missing_cols = [c for c in (idx_col, window_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV '{csv_path}' is missing columns: {missing_cols}\n"
            f"Re-run video_index.py to generate an enriched CSV with frame indices."
        )

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(emb_cache_dir)

    for _, row in df.iterrows():
        video_path = row["video_path"]
        subset = row["subset"]
        source_model = row["source_model"]
        filename = Path(video_path).name

        # Parse stored JSON index lists
        window_idxs_raw = row[window_col]
        if _is_missing_window(window_idxs_raw):
            # Video too short for this duration — skip
            continue
        downsample_idxs = json.loads(row[idx_col])
        window_idxs = json.loads(window_idxs_raw)

        # Resolve cache paths
        stem = Path(video_path).stem
        compact_cache_path = _get_cache_path(cache_root, subset, source_model, stem, duration_sec, compact=True)
        full_cache_path = _get_cache_path(cache_root, subset, source_model, stem, duration_sec, compact=False)

        if compact and compact_cache_path.exists():
            # Compact cache: already contains exactly the window frames, no re-indexing needed
            emb = torch.load(compact_cache_path, weights_only=True).numpy()  # (T_window, D)
        elif full_cache_path.exists():
            full_emb = torch.load(full_cache_path, weights_only=True).numpy()  # (N_8fps, D)
            emb = _slice_window(full_emb, downsample_idxs, window_idxs, video_path)
            if emb is None:
                continue
        else:
            # Cache miss: extract embeddings for the downsampled frames
            frames = load_video_frames(video_path, downsample_idxs)
            full_emb = model.frames_to_embeddings([frames], batch_size=batch_size)[0]
            full_cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write via temp file to avoid partial .pt files
            tmp = full_cache_path.with_suffix(".tmp.pt")
            torch.save(torch.from_numpy(full_emb), tmp)
            tmp.rename(full_cache_path)
            emb = _slice_window(full_emb, downsample_idxs, window_idxs, video_path)
            if emb is None:
                continue

        yield {
            "embs": emb[np.newaxis],  # [1, T, D]
            "subset": subset,
            "source_model": source_model,
            "filename": filename,
        }
