"""
organize_videofeedback.py

Copy flat VideoFeedback zip extractions into the generator-labeled directory
structure expected by video_index.py / eval.py.

Usage:
    python src/organize_videofeedback.py \
        --tmp-dir datasets/videofeedback_tmp \
        --output  datasets/videofeedback

The --tmp-dir is the root of the cloned TIGER-Lab/VideoFeedback repo (with
extracted/ subdirs from unzipping the four zip files).  The script reads the
four JSON metadata files to map each video ID to its source model, then copies
videos to --output/{real|fake}/{source_model}/{id}.mp4.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

SOURCE_MODEL_MAP = {
    "vidprom_pika":         "Pika",
    "vidprom_t2vz":         "Text2Video-Zero",
    "vidprom_vc2":          "VideoCrafter2",
    "vidprom_ms":           "ModelScope",
    "vidprom_lavie_base":   "LaVie-base",
    "vidprom_anidiff":      "AnimateDiff",
    "vidprom_lvdm":         "LVDM",
    "vidprom_hotshot":      "Hotshot-XL",
    "vidprom_zs_576w":      "ZeroScope-576w",
    "fastsvd":              "Fast-SVD",
    "sora":                 "SoRA-Clip",
    "real_didemo_high_res": "DiDeMo",
    "real_pd70m_high_res":  "Panda70M",
}


def organize(tmp_dir: Path, output: Path) -> None:
    splits = ["train", "test"]
    subsets = ["real", "annotated"]

    counts: Counter = Counter()
    unmapped: Counter = Counter()
    missing: list[str] = []

    # Load all entries first so tqdm knows the total upfront.
    all_items: list[tuple] = []  # (item, top_dir, extracted_dir)
    for split in splits:
        for subset in subsets:
            json_path = tmp_dir / split / f"data_{subset}.json"
            if not json_path.exists():
                print(f"[warn] missing JSON: {json_path}")
                continue

            with open(json_path) as f:
                entries = json.load(f)

            top_dir = "real" if subset == "real" else "fake"
            extracted_dir = tmp_dir / "extracted" / f"{subset}_{split}"
            for item in entries:
                all_items.append((item, top_dir, extracted_dir))

    for item, top_dir, extracted_dir in tqdm(all_items, desc="Moving videos", unit="video"):
        vid_id = item.get("id", "")
        video_url = item.get("video link", "")
        path_parts = urlparse(video_url).path.strip("/").split("/")
        raw_model = path_parts[-2] if len(path_parts) >= 2 else ""
        source_model = SOURCE_MODEL_MAP.get(raw_model)

        if source_model is None:
            unmapped[raw_model] += 1
            continue

        src = extracted_dir / f"{vid_id}.mp4"
        dest = output / top_dir / source_model / f"{vid_id}.mp4"
        if dest.exists():
            counts[f"{top_dir}/{source_model}"] += 1
            continue
        if not src.exists():
            missing.append(str(src))
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
        counts[f"{top_dir}/{source_model}"] += 1

    print("\nMoved videos per generator:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"\nTotal: {sum(counts.values())}")

    if unmapped:
        print("\n[warn] unmapped raw_source_model values (extend SOURCE_MODEL_MAP):")
        for raw, n in sorted(unmapped.items()):
            print(f"  {raw!r}: {n} videos")

    if missing:
        print(f"\n[warn] {len(missing)} source files not found (first 10):")
        for p in missing[:10]:
            print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize VideoFeedback into generator subfolders.")
    parser.add_argument("--tmp-dir", required=True, type=Path,
                        help="Root of the cloned TIGER-Lab/VideoFeedback repo (with extracted/ subdirs)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Target dataset directory (e.g. datasets/videofeedback)")
    args = parser.parse_args()

    organize(args.tmp_dir, args.output)


if __name__ == "__main__":
    main()
