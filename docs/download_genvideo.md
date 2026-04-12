# Downloading the GenVideo Dataset

GenVideo contains ~8k videos across 10 AI video generators (fake) and MSR-VTT real videos.


### Prerequisites

```bash
pip install modelscope kaggle
```

**Kaggle credentials** (required for the real videos step):
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

---

### Step 1 — Download fake videos (GenVideo-Val.zip, ~13 GB)

```bash
mkdir -p datasets/genvideo/fake datasets/genvideo/real
modelscope download --dataset cccnju/Gen-Video GenVideo-Val.zip \
    --local_dir datasets/genvideo/
```

**Verify:**
```bash
md5sum datasets/genvideo/GenVideo-Val.zip
# expected: 812e14ee004a32e5bf0e14082065f03d
```

---

### Step 2 — Unzip

```bash
unzip datasets/genvideo/GenVideo-Val.zip -d datasets/genvideo/tmp/
```

**Verify:**
```bash
ls datasets/genvideo/tmp/GenVideo-Val/Fake/
# expected: Crafter  Gen2  HotShot  Lavie  ModelScope  MoonValley  MorphStudio  Show_1  Sora  WildScrape
```

---

### Step 3 — Move fake videos into place

```bash
mv datasets/genvideo/tmp/GenVideo-Val/Fake/* datasets/genvideo/fake/
```

**Verify:**
```bash
ls datasets/genvideo/fake/
# expected: Crafter  Gen2  HotShot  Lavie  ModelScope  MoonValley  MorphStudio  Show_1  Sora  WildScrape
```

If everything looks correct, clean up:
```bash
rm -rf datasets/genvideo/tmp datasets/genvideo/GenVideo-Val.zip
```

---

### Step 4 — Download real MSR-VTT videos from Kaggle

```bash
kaggle datasets download -d khoahunhtngng/msrvtt \
    --path datasets/genvideo/real/MSR-VTT --unzip
```

Kaggle extracts into a nested `MSR-VTT/MSR-VTT/` folder with `TrainValVideo/` and `TestVideo/` subdirs. Flatten everything into `MSR-VTT/` directly:

```bash
find datasets/genvideo/real/MSR-VTT/MSR-VTT/TrainValVideo \
     datasets/genvideo/real/MSR-VTT/MSR-VTT/TestVideo \
     -name "*.mp4" -exec mv {} datasets/genvideo/real/MSR-VTT/ \;
```

**Verify:**
```bash
ls datasets/genvideo/real/MSR-VTT/*.mp4 | wc -l
# expected: 10000
```

If everything looks correct, clean up:
```bash
# Remove everything except the .mp4 files
find datasets/genvideo/real/MSR-VTT -mindepth 1 -not -name "*.mp4" -exec rm -rf {} +
```

---

## Final directory layout

```
datasets/genvideo/
├── real/
│   └── MSR-VTT/       *.mp4
└── fake/
    ├── Crafter/        *.mp4
    ├── Gen2/           *.mp4
    ├── HotShot/        *.mp4
    ├── Lavie/          *.mp4
    ├── ModelScope/     *.mp4
    ├── MoonValley/     *.mp4
    ├── MorphStudio/    *.mp4
    ├── Show_1/         *.mp4
    ├── Sora/           *.mp4
    └── WildScrape/     *.mp4
```

Once done, return to the [README](../README.md#-reproducing-paper-results) to run STALL on GenVideo.
