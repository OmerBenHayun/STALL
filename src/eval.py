"""Phase 4 eval CLI for STALL.

Usage:
    # HuggingFace dataset — uses pre-computed embeddings, no DINOv3 needed
    python eval.py --hf-dataset OmerXYZ/comgenvid

    # CSV with columns: video_path, subset, source_model
    python eval.py --csv my_benchmark.csv

    # Enriched CSV from video_index.py — uses embedding cache
    python src/eval.py --csv cache/indexes/genvideo.csv --emb-cache cache/embeddings/genvideo/ --output-csv results.csv

    # Two directories containing <model>/*.mp4 subdirs
    python src/eval.py --real-dir datasets/demo_dataset/real/ --fake-dir datasets/demo_dataset/fake/

    # Save per-video scores to CSV
    python src/eval.py --hf-dataset OmerXYZ/comgenvid --output-csv results.csv

    # Override DINOv3 paths (only needed for CSV/dir modes)
    python src/eval.py --csv bench.csv --dino-repo ~/dinov3 --dino-weights ~/dinov3/weights/...pth

    # Debug: 5 videos per source
    python src/eval.py --csv cache/indexes/genvideo.csv --emb-cache cache/embeddings/genvideo/ --debug-n 5 --output-csv dbg.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from metrics import Score, ScoreDirection, get_results_df, print_results
from stall import STALL

STALL_PARAMS_DEFAULT = "precomputed/stall_params_vatex_dino_v3.npz"


# ─────────────────────────────────────────────────────────────────────────────
# STALL factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_stall(args, load_dino: bool) -> STALL:
    data = np.load(args.params)
    device = "cpu"
    if load_dino:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
    return STALL(
        device=device,
        data_dict=data,
        load_dino=load_dino,
        dino_repo=getattr(args, "dino_repo", None),
        dino_weights=getattr(args, "dino_weights", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-mode runners
# ─────────────────────────────────────────────────────────────────────────────

def _print_debug_pair(model, samples: dict):
    """Print per-step debug output for one real and one fake sample."""
    SEP = "=" * 60
    for label, subset_key in [("REAL", "real"), ("FAKE", "annotated")]:
        sample = samples.get(subset_key)
        if sample is None:
            print(f"\n{SEP}\n  {label}: (no sample found)\n{SEP}")
            continue
        print(f"\n{SEP}")
        print(f"  {label}: {sample.get('filename', '?')}  [{sample.get('source_model', '?')}]")
        print(SEP)
        result = model._scores_from_embs(sample["embs"])
        model.print_score_debug(result)
    print(f"\n{SEP}")
    print("  Expected for correct HIGHER_IS_REAL behavior:")
    print("    real final_score  >  fake final_score")
    print(f"{SEP}\n")


def run_hf(args) -> pd.DataFrame:
    from dataset_utils import load_hf_dataset
    from tqdm import tqdm

    print(f"Loading HuggingFace dataset: {args.hf_dataset}")
    model = _make_stall(args, load_dino=False)

    debug = getattr(args, "debug", False)
    _debug_samples = {}  # subset -> sample, populated on first occurrence of each

    rows = []
    for sample in tqdm(load_hf_dataset(args.hf_dataset, split=args.split, duration=getattr(args, "duration", 2), verbose=args.debug), desc="Scoring", unit="video", dynamic_ncols=True):
        result = model._scores_from_embs(sample["embs"])
        rows.append({
            "subset": sample["subset"],
            "source_model": sample["source_model"],
            "filename": sample["filename"],
            "final_score": float(result["final_score"][0]),
        })
        if debug and sample["subset"] not in _debug_samples:
            _debug_samples[sample["subset"]] = sample

    if debug:
        _print_debug_pair(model, _debug_samples)

    return pd.DataFrame(rows)


def run_csv(args) -> pd.DataFrame:
    from dataset_utils import load_csv

    # Peek at the CSV to decide which path to take
    df_peek = load_csv(args.csv)
    emb_cache = getattr(args, "emb_cache", None)
    duration_sec = getattr(args, "duration", 2)
    debug_n = getattr(args, "debug_n", None)
    debug = getattr(args, "debug", False)

    use_cache_path = emb_cache is not None or f"{duration_sec}_sec_idxs" in df_peek.columns

    if use_cache_path:
        # Enriched CSV path: per-video embedding cache + frame-index slicing
        from dataset_utils import count_cache_misses, load_csv_with_emb_cache, prefill_emb_cache
        from tqdm import tqdm

        model = _make_stall(args, load_dino=True)
        _csv_df = pd.read_csv(args.csv)
        if debug_n is not None:
            _total = _csv_df.groupby(["subset", "source_model"]).size().clip(upper=debug_n).sum()
        else:
            _total = len(_csv_df)

        # Phase 1: extract DINOv3 embeddings (parallel decode + cross-video GPU batching)
        _n_misses = count_cache_misses(
            args.csv,
            emb_cache_dir=emb_cache or ".",
            duration_sec=duration_sec,
            debug_n=debug_n,
            compact=args.compact,
        )
        print(f"Phase 1/2 — Extracting DINOv3 embeddings ({_n_misses} cache misses, {_total - _n_misses} cached)")
        for _ in tqdm(
            prefill_emb_cache(
                args.csv,
                emb_cache_dir=emb_cache or ".",
                model=model,
                duration_sec=duration_sec,
                debug_n=debug_n,
                compact=args.compact,
                num_workers=args.workers,
                video_batch=args.video_batch,
            ),
            desc="Extracting", unit=" video", dynamic_ncols=True, total=_n_misses,
        ):
            pass

        # Phase 2: score all videos from cache
        print("Phase 2/2 — Scoring")
        _debug_samples = {}
        rows = []
        for sample in tqdm(
            load_csv_with_emb_cache(
                args.csv,
                emb_cache_dir=emb_cache or ".",
                model=model,
                duration_sec=duration_sec,
                debug_n=debug_n,
                compact=args.compact,
            ),
            desc="Scoring", unit="video", dynamic_ncols=True, total=_total,
        ):
            result = model._scores_from_embs(sample["embs"])
            rows.append({
                "subset": sample["subset"],
                "source_model": sample["source_model"],
                "filename": sample["filename"],
                "final_score": float(result["final_score"][0]),
            })
            if debug and sample["subset"] not in _debug_samples:
                _debug_samples[sample["subset"]] = sample

        if debug:
            _print_debug_pair(model, _debug_samples)

        return pd.DataFrame(rows)

    else:
        # Backward-compatible path: plain CSV, on-the-fly batch inference
        df = df_peek
        if debug_n is not None:
            df = (
                df.groupby(["subset", "source_model"], group_keys=False)
                .apply(lambda g: g.head(debug_n))
                .reset_index(drop=True)
            )
        print(f"Loaded {len(df)} videos from {args.csv}")
        model = _make_stall(args, load_dino=True)
        results = model.batch_inference(df["video_path"].tolist())
        df = df.copy()
        df["final_score"] = [float(r["final_score"][0]) for r in results]

        if debug:
            _debug_samples = {}
            for i, (_, row) in enumerate(df.iterrows()):
                subset = row["subset"]
                if subset not in _debug_samples:
                    _debug_samples[subset] = {
                        "embs": results[i]["embs"],
                        "filename": row.get("filename", row["video_path"]),
                        "source_model": row["source_model"],
                    }
            _print_debug_pair(model, _debug_samples)

        return df


def run_dirs(args) -> pd.DataFrame:
    from pathlib import Path

    records = []
    for dir_path, subset_val in [(args.real_dir, "real"), (args.fake_dir, "annotated")]:
        root = Path(dir_path)
        if not root.exists():
            raise ValueError(f"Directory not found: {dir_path}")
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            for video_file in sorted(model_dir.glob("*.mp4")):
                records.append({
                    "video_path": str(video_file),
                    "subset": subset_val,
                    "source_model": model_dir.name,
                })

    if not records:
        raise ValueError("No .mp4 files found in the specified directories.")

    df = pd.DataFrame(records)
    n_real = (df["subset"] == "real").sum()
    n_fake = (df["subset"] == "annotated").sum()
    print(f"Found {len(df)} videos: {n_real} real, {n_fake} fake")

    model = _make_stall(args, load_dino=True)
    results = model.batch_inference(df["video_path"].tolist())
    df["final_score"] = [float(r["final_score"][0]) for r in results]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run STALL detector and print per-generator AUC/AP results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hf-dataset", metavar="REPO_ID",
        help="HuggingFace repo ID (uses pre-computed embeddings — no DINOv3 needed)",
    )
    src.add_argument(
        "--csv", metavar="CSV_PATH",
        help="CSV with columns: video_path, subset ('real'/'annotated'), source_model",
    )
    src.add_argument(
        "--real-dir", metavar="DIR",
        help="Directory of real videos: <model>/*.mp4 subdirs (pair with --fake-dir)",
    )

    parser.add_argument(
        "--fake-dir", metavar="DIR",
        help="Directory of fake videos: <model>/*.mp4 subdirs (pair with --real-dir)",
    )
    parser.add_argument(
        "--dino-repo", metavar="PATH", default=None,
        help="Override path to local DINOv3 repo clone (CSV/dir modes only)",
    )
    parser.add_argument(
        "--dino-weights", metavar="PATH", default=None,
        help="Override path to DINOv3 .pth weights file (CSV/dir modes only)",
    )
    parser.add_argument(
        "--params", default=STALL_PARAMS_DEFAULT,
        help=f"STALL params .npz (default: {STALL_PARAMS_DEFAULT})",
    )
    parser.add_argument(
        "--output-csv", metavar="PATH", default=None,
        help="Save per-video scores (subset, source_model, final_score, …) to CSV",
    )
    parser.add_argument(
        "--split", default="train",
        help="HuggingFace dataset split (default: train)",
    )
    parser.add_argument(
        "--debug", nargs="?", const=True, default=None, metavar="N",
        help=(
            "Enable debug output: print per-step scores for one real and one fake sample. "
            "Optionally pass N to also limit scoring to N videos per (subset, source_model), "
            "e.g. --debug 5."
        ),
    )
    parser.add_argument(
        "--emb-cache", metavar="PATH", default=None,
        help=(
            "Directory for DINOv3 embedding cache (.pt files per video). "
            "If omitted, compute embeddings on-the-fly without caching. "
            "Used with enriched CSVs produced by video_index.py."
        ),
    )
    parser.add_argument(
        "--duration", type=int, default=2, choices=[1, 2, 3, 4],
        help="Which second-window to use for scoring (default: 2). Requires enriched CSV.",
    )
    parser.add_argument(
        "--compact", action="store_true", default=False,
        help=(
            "Extract only the --duration-second window frames instead of the full "
            "video at 8 fps. Cached as {stem}_{duration}s.pt. Greatly speeds up "
            "extraction for long videos. To re-score with a different --duration "
            "later, re-run without --compact to build a full cache. "
            "(CSV + --emb-cache mode only)"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="CPU decode threads for parallel video loading in Phase 1 (default: 4)",
    )
    parser.add_argument(
        "--video-batch", type=int, default=8, metavar="N",
        help="Videos batched together for a single GPU pass in Phase 1 (default: 8)",
    )

    args = parser.parse_args()

    # Unpack --debug [N] into separate bool and int for use throughout
    _debug_raw = args.debug
    args.debug = _debug_raw is not None
    args.debug_n = int(_debug_raw) if isinstance(_debug_raw, str) else None

    if args.real_dir and not args.fake_dir:
        parser.error("--real-dir requires --fake-dir")
    if args.fake_dir and not args.real_dir:
        parser.error("--fake-dir requires --real-dir")

    # Run appropriate loader
    if args.hf_dataset:
        df = run_hf(args)
    elif args.csv:
        df = run_csv(args)
    else:
        df = run_dirs(args)

    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved per-video scores → {args.output_csv}")

    scores_d = {
        "final_score": Score(
            value=df["final_score"].to_numpy(),
            direction=ScoreDirection.HIGHER_IS_REAL,
        )
    }
    results_df = get_results_df(df[["subset", "source_model"]], scores_d)
    print_results(results_df)


if __name__ == "__main__":
    main()
