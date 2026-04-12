# Downloading the VideoFeedback Dataset

VideoFeedback contains ~8.5k videos from 11 AI video generators (fake) and 2 real video sources (DiDeMo, Panda70M).

### Prerequisites

```bash
pip install huggingface_hub hf_transfer
```

---

### Step 1 — Download

`hf_transfer` enables multi-part parallel downloads and is significantly faster than the default HF client:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download TIGER-Lab/VideoFeedback \
    --repo-type dataset \
    --include "train/videos_annotated.zip" \
    --include "train/videos_real.zip" \
    --include "test/videos_annotated.zip" \
    --include "test/videos_real.zip" \
    --include "train/data_annotated.json" \
    --include "train/data_real.json" \
    --include "test/data_annotated.json" \
    --include "test/data_real.json" \
    --local-dir datasets/videofeedback_tmp
```

---

### Step 2 — Unzip

```bash
mkdir -p datasets/videofeedback_tmp/extracted
pushd datasets/videofeedback_tmp
unzip train/videos_annotated.zip -d extracted/annotated_train/
unzip train/videos_real.zip      -d extracted/real_train/
unzip test/videos_annotated.zip  -d extracted/annotated_test/
unzip test/videos_real.zip       -d extracted/real_test/
popd
```

---

### Step 3 — Organize into generator subfolders

The zip files contain flat `{id}.mp4` files with no generator info in the filename. The JSON metadata files map each video ID to its source model. Run the organize script to copy videos into the correct subfolders:

```bash
python src/organize_videofeedback.py \
    --tmp-dir datasets/videofeedback_tmp \
    --output  datasets/videofeedback
```

The script prints per-generator counts on completion.

---

### Step 4 — Clean up

```bash
rm -rf datasets/videofeedback_tmp
```

---

## Final directory layout

```
datasets/videofeedback/
├── real/
│   ├── DiDeMo/            *.mp4
│   └── Panda70M/          *.mp4
└── fake/
    ├── AnimateDiff/        *.mp4
    ├── Fast-SVD/           *.mp4
    ├── Hotshot-XL/         *.mp4
    ├── LaVie-base/         *.mp4
    ├── LVDM/               *.mp4
    ├── ModelScope/         *.mp4
    ├── Pika/               *.mp4
    ├── SoRA-Clip/          *.mp4
    ├── Text2Video-Zero/    *.mp4
    ├── VideoCrafter2/      *.mp4
    └── ZeroScope-576w/     *.mp4
```

Once done, return to the [README](../README.md#-reproducing-paper-results) to run STALL on VideoFeedback.
