# Downloading the ComGenVid Dataset

ComGenVid contains ~3.5k generated videos from two frontier models (Sora and VEO3) paired with ~1.7k real videos from MSVD.

### Prerequisites

```bash
# git-lfs is required to download the video files
git lfs install
```

On Ubuntu/Debian:
```bash
sudo apt-get install git-lfs
```

On macOS:
```bash
brew install git-lfs
```

---

### Download

Clone the dataset without LFS files first (fast), then pull only the video files:

```bash
# Clone repo structure without downloading LFS content yet
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/OmerXYZ/comgenvid datasets/comgenvid_tmp

# Pull only the video files via LFS batch transfer
cd datasets/comgenvid_tmp
git lfs pull --include="videos/**"
cd ../..
```

> Using git LFS batch transfer avoids the HuggingFace API rate limit (1000 requests/5 min) that affects `hf download` when downloading many files.

---

### Move into place

```bash
mv datasets/comgenvid_tmp/videos datasets/comgenvid
rm -rf datasets/comgenvid_tmp
```

---

### Verify

```bash
ls datasets/comgenvid/real/MSVD/*.mp4 | wc -l
# expected: ~1700

ls datasets/comgenvid/fake/Sora/*.mp4 | wc -l
# expected: ~1700

ls datasets/comgenvid/fake/VEO3/*.mp4 | wc -l
# expected: ~1700
```

---

## Final directory layout

```
datasets/comgenvid/
├── real/
│   └── MSVD/          *.mp4
└── fake/
    ├── Sora/           *.mp4
    └── VEO3/           *.mp4
```

Once done, return to the [README](../README.md#-reproducing-paper-results) to run STALL on ComGenVid.
