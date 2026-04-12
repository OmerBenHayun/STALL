import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DINO_V3_MODEL_NAME = "dinov3_vitl16"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default paths — override via DINO_V3_REPO_DIR / DINO_V3_WEIGHTS env vars
DINO_V3_REPO_DIR = os.getenv(
    "DINO_V3_REPO_DIR", os.path.join(_REPO_ROOT, "dinov3")
)
DINO_V3_WEIGHTS = os.getenv(
    "DINO_V3_WEIGHTS",
    os.path.join(_REPO_ROOT, "dinov3", "weights", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
)

DINOV3_GITHUB_URL = "https://github.com/facebookresearch/dinov3"

AGG_STR2FN = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
}


# ─────────────────────────────────────────────────────────────────────────────
# DINOv3
# ─────────────────────────────────────────────────────────────────────────────

def create_dinov3_transform(resize_size: int = 224):
    """Standard ImageNet eval transform for DINOv3 models pretrained on LVD-1689M."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_dinov3_model(device: str, repo_dir: str = None, weights: str = None):
    """Load DINOv3 ViT-L/16 model from a local clone of the dinov3 repo.

    Args:
        device:    Target device string ("cuda" / "cpu").
        repo_dir:  Path to local dinov3 repo clone. Falls back to DINO_V3_REPO_DIR.
        weights:   Path to .pth weights file. Falls back to DINO_V3_WEIGHTS.

    Returns:
        (model, transform)
    """
    repo_dir = repo_dir or DINO_V3_REPO_DIR
    weights = weights or DINO_V3_WEIGHTS

    if not os.path.exists(repo_dir):
        raise ValueError(
            f"DINOv3 repo not found at '{repo_dir}'.\n"
            f"Clone it from {DINOV3_GITHUB_URL} and place the weights in weights/.\n"
            f"Override the path via --dino-repo or the DINO_V3_REPO_DIR env var."
        )
    if not os.path.exists(weights):
        raise ValueError(
            f"DINOv3 weights not found at '{weights}'.\n"
            f"Download dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth from {DINOV3_GITHUB_URL}.\n"
            f"Override the path via --dino-weights or the DINO_V3_WEIGHTS env var."
        )

    # Import the backbone directly instead of via torch.hub.load, which would
    # import hubconf.py and drag in detectors/segmentors that require
    # torchvision.transforms.v2 (torchvision >= 0.15).
    import sys
    sys.path.insert(0, repo_dir)
    try:
        from dinov3.hub.backbones import dinov3_vitl16
        model = dinov3_vitl16(weights=weights)
    finally:
        if repo_dir in sys.path:
            sys.path.remove(repo_dir)
    model = model.to(device).eval()
    transform = create_dinov3_transform()
    print(f"DINOv3 model loaded on {device}")
    return model, transform


# ─────────────────────────────────────────────────────────────────────────────
# Math utilities
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood(array):
    """Gaussian log-likelihood under N(0,I). Input shape: [N, T, D]. Returns [N, T]."""
    D = array.shape[-1]
    return -0.5 * (D * np.log(2.0 * np.pi) + (array ** 2).sum(axis=-1))


def whitening_transform(emb, mu, W):
    """Apply pre-fitted whitening: (emb - mu) @ W."""
    return np.matmul(emb - mu, W)


def raw_emb_to_log_likelihoods(raw_emb, mu, W, preprocess_fn=None):
    if preprocess_fn is not None:
        raw_emb = preprocess_fn(raw_emb)
    return log_likelihood(whitening_transform(raw_emb, mu, W))


def diff_vec(arr):
    """Consecutive frame differences. [N, T, D] → [N, T-1, D]."""
    return arr[:, 1:, :] - arr[:, :-1, :]


def l2_normalize(arr):
    arr = arr.astype(np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return arr / norm


def diff_normalized_embeddings(arr):
    """L2-normalized consecutive frame embedding differences. [N,T,D] → [N,T-1,D]."""
    return l2_normalize(diff_vec(arr))


def get_percentile_score(inf_scores, sorted_calib_scores):
    """Fraction of calibration scores <= each inference score. Both 1-D arrays."""
    positions = np.searchsorted(sorted_calib_scores, inf_scores, side="right")
    return positions / len(sorted_calib_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_video_frames(video_path, frame_indices=None):
    """Load frames from an MP4 file.

    Args:
        video_path:    Path to .mp4 file.
        frame_indices: Optional list of 0-based frame indices to load.
                       If None, all frames are loaded sequentially.

    Returns:
        np.ndarray of shape [T, H, W, C] in BGR order.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if frame_indices is None:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    frame_indices_sorted = sorted(frame_indices)
    frames_dict = {}
    for idx in frame_indices_sorted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_dict[idx] = frame
    cap.release()
    return np.array([frames_dict[i] for i in frame_indices if i in frames_dict])


# ─────────────────────────────────────────────────────────────────────────────
# STALL detector
# ─────────────────────────────────────────────────────────────────────────────

class STALL:
    """Spatial-Temporal Anomaly Log-Likelihood detector for AI-generated videos.

    Higher ``final_score`` → more likely real (score direction: HIGHER_IS_REAL).

    Args:
        device:     Torch device string.
        data_dict:  Dict / npz with keys W_spat, mu_spat, W_temp, mu_temp,
                    calib_ll_spat, calib_ll_temp.
        spat_agg:   Aggregation over frames for the spatial branch ("max").
        temp_agg:   Aggregation over frame-pairs for the temporal branch ("min").
        dino_repo:  Optional override for DINOv3 repo path.
        dino_weights: Optional override for DINOv3 weights path.
    """

    # Class-level cache so multiple STALL instances share one loaded model.
    _shared_model = None
    _shared_transform = None

    def __init__(
        self,
        device,
        data_dict: dict,
        spat_agg: str = "max",
        temp_agg: str = "min",
        dino_repo: str = None,
        dino_weights: str = None,
        load_dino: bool = True,
    ):
        if load_dino:
            if STALL._shared_model is None:
                STALL._shared_model, STALL._shared_transform = load_dinov3_model(
                    device, repo_dir=dino_repo, weights=dino_weights
                )
            self.model = STALL._shared_model
            self.transform = STALL._shared_transform
        else:
            self.model = None
            self.transform = None

        self.device = device

        self.w_spat = data_dict["W_spat"]
        self.mu_spat = data_dict["mu_spat"]
        self.w_temp = data_dict["W_temp"]
        self.mu_temp = data_dict["mu_temp"]

        self.spat_agg = spat_agg
        self.temp_agg = temp_agg

        calib_spat = data_dict["calib_ll_spat"]
        calib_temp = data_dict["calib_ll_temp"]
        self.calib_spat_sorted = np.sort(AGG_STR2FN[spat_agg](calib_spat, axis=1))
        self.calib_temp_sorted = np.sort(AGG_STR2FN[temp_agg](calib_temp, axis=1))

    # ── Embedding extraction ──────────────────────────────────────────────────

    def _embed_flat_frames(self, flat_frames: list, batch_size: int) -> np.ndarray:
        """Run DINOv3 on a flat list of BGR frames. Returns [total_frames, D]."""
        device = next(self.model.parameters()).device
        flat_embs = []
        with torch.no_grad():
            for start in range(0, len(flat_frames), batch_size):
                batch = flat_frames[start : start + batch_size]
                tensors = [
                    self.transform(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                    for fr in batch
                ]
                flat_embs.append(self.model(torch.stack(tensors).to(device)))
        return torch.cat(flat_embs, dim=0).cpu().numpy()

    def frames_to_embeddings(self, video_arrays, batch_size=32):
        """Extract DINOv3 embeddings for a list of videos.

        Args:
            video_arrays: List of np.ndarray, each [T, H, W, C] BGR.
            batch_size:   Frames per GPU batch.

        Returns:
            np.ndarray of shape [N, T, D].
        """
        lengths = [len(v) for v in video_arrays]
        flat_frames = [frame for video in video_arrays for frame in video]
        flat_embs = self._embed_flat_frames(flat_frames, batch_size)
        cursor = 0
        all_embeddings = []
        for length in lengths:
            all_embeddings.append(flat_embs[cursor : cursor + length])
            cursor += length
        return np.stack(all_embeddings)

    # ── Branch log-likelihoods ────────────────────────────────────────────────

    def spatial_log_likelihood(self, embs):
        """Spatial LL: whitened raw embeddings → N(0,I). Shape [N, T]."""
        return raw_emb_to_log_likelihoods(embs, mu=self.mu_spat, W=self.w_spat)

    def temporal_log_likelihood(self, embs):
        """Temporal LL: whitened L2-normalized frame diffs → N(0,I). Shape [N, T-1]."""
        zero_mask = (np.linalg.norm(diff_vec(embs), axis=-1) == 0)  # [N, T-1]
        ll = raw_emb_to_log_likelihoods(
            embs, mu=self.mu_temp, W=self.w_temp, preprocess_fn=diff_normalized_embeddings
        )
        # Identical consecutive frames produce a zero diff with no temporal information.
        # Setting their LL to +inf excludes them from the min aggregation -- they never
        # win. NOTE: if you change temp_agg from "min" to "mean" or "max", revisit
        # this -- +inf would corrupt those aggregations and you would need a different
        # strategy.
        ll[zero_mask] = np.inf
        return ll

    # ── Shared scoring logic ──────────────────────────────────────────────────

    def _scores_from_embs(self, embs):
        """Compute all STALL scores from a [1, T, D] embedding array.

        Returns the same dict structure as inference().
        """
        spat_ll = self.spatial_log_likelihood(embs)
        temp_ll = self.temporal_log_likelihood(embs)

        spat_agg = AGG_STR2FN[self.spat_agg](spat_ll, axis=1)
        temp_agg = AGG_STR2FN[self.temp_agg](temp_ll, axis=1)

        spat_pct = get_percentile_score(spat_agg, self.calib_spat_sorted)
        temp_pct = get_percentile_score(temp_agg, self.calib_temp_sorted)

        return {
            "embs": embs,
            "spat_ll": spat_ll,
            "temp_ll": temp_ll,
            "spat_ll_agg": spat_agg,
            "temp_ll_agg": temp_agg,
            "spat_percentile": spat_pct,
            "temp_percentile": temp_pct,
            "final_score": 0.5 * (spat_pct + temp_pct),
        }

    # ── Single-video inference ────────────────────────────────────────────────

    def inference(self, video_path, frame_indices=None):
        """Run STALL on a single video file.

        Returns a dict with keys:
            embs, spat_ll, temp_ll,
            spat_ll_agg, temp_ll_agg,
            spat_percentile, temp_percentile,
            final_score   (0-1, higher = more likely real, HIGHER_IS_REAL)
        """
        frames = load_video_frames(video_path, frame_indices)
        embs = self.frames_to_embeddings([frames])
        return self._scores_from_embs(embs)

    # ── Batch inference ───────────────────────────────────────────────────────

    def batch_inference(self, video_paths, frame_indices_list=None, batch_size=32):
        """Run STALL on multiple videos in one efficient pass.

        All frames from all videos are flattened into a single sequence and
        processed through DINOv3 together (``batch_size`` frames per forward
        pass), then split back into per-video embeddings. Scores are computed
        independently per video, so results are numerically identical to calling
        ``inference()`` on each video separately.

        Args:
            video_paths:        List of paths to .mp4 files.
            frame_indices_list: Optional list of frame-index lists, one per
                                video. ``None`` entries (or omitting the arg
                                entirely) load all frames for that video.
            batch_size:         Frames per DINOv3 forward pass.

        Returns:
            List of result dicts in the same format as ``inference()``,
            one per input video, in the same order.
        """
        if frame_indices_list is None:
            frame_indices_list = [None] * len(video_paths)

        # 1. Load all frames ──────────────────────────────────────────────────
        all_frames = [
            load_video_frames(path, fidx)
            for path, fidx in zip(video_paths, frame_indices_list)
        ]
        lengths = [len(f) for f in all_frames]          # frames per video
        flat_frames = [frame for vid in all_frames for frame in vid]

        # 2. Single batched DINOv3 pass over all frames ───────────────────────
        flat_embs = self._embed_flat_frames(flat_frames, batch_size)  # [total_frames, D]

        # 3. Split embeddings back per video and score ────────────────────────
        results = []
        cursor = 0
        for length in lengths:
            embs = flat_embs[cursor : cursor + length][np.newaxis]  # [1, T, D]
            cursor += length
            results.append(self._scores_from_embs(embs))
        return results

    # ── Debug output ──────────────────────────────────────────────────────────

    def print_score_debug(self, result: dict):
        """Print intermediate scoring values from an inference result dict.

        Pass the dict returned by inference() or _scores_from_embs() to verify:
          - Log-likelihoods are negative (as expected for Gaussian LL)
          - Whitened embeddings have unit variance per dimension
          - Aggregated LL position relative to calibration range
          - Percentile scores (real should be above ~0.5, fake below)
        """
        embs    = result["embs"]
        spat_ll = result["spat_ll"]
        temp_ll = result["temp_ll"]
        spat_agg = result["spat_ll_agg"]
        temp_agg = result["temp_ll_agg"]
        spat_pct = result["spat_percentile"]
        temp_pct = result["temp_percentile"]

        D = embs.shape[-1]
        w_embs_spat = whitening_transform(embs, self.mu_spat, self.w_spat)
        w_embs_temp = whitening_transform(
            diff_normalized_embeddings(embs), self.mu_temp, self.w_temp
        )
        print(f"  [debug] emb shape:           {embs.shape}  (D={D})")
        print(f"  [debug] whitened_spat norm²  mean={np.mean(w_embs_spat**2):.3f}  (expected ~1.0 per dim)")
        print(f"  [debug] whitened_temp norm²  mean={np.mean(w_embs_temp**2):.3f}  (expected ~1.0 per dim)")
        print(f"  [debug] spat_ll per-frame:   min={spat_ll.min():.1f}  max={spat_ll.max():.1f}  mean={spat_ll.mean():.1f}")
        print(f"  [debug] temp_ll per-pair:    min={temp_ll.min():.1f}  max={temp_ll.max():.1f}  mean={temp_ll.mean():.1f}")
        print(f"  [debug] spat_ll_agg ({self.spat_agg}):  {spat_agg}  |  calib range [{self.calib_spat_sorted[0]:.1f}, {self.calib_spat_sorted[-1]:.1f}]")
        print(f"  [debug] temp_ll_agg ({self.temp_agg}):  {temp_agg}  |  calib range [{self.calib_temp_sorted[0]:.1f}, {self.calib_temp_sorted[-1]:.1f}]")
        print(f"  [debug] spat_percentile:     {spat_pct}")
        print(f"  [debug] temp_percentile:     {temp_pct}")
        print(f"  [debug] final_score:         {0.5 * (spat_pct + temp_pct)}")
