"""
Shared machinery for the calibrated H&E color pipeline
(VirtualH&E_semi-supervised_cycleGAN_30_08_26).

This is a fresh, self-contained module -- it does not import from or depend
on the existing project (VirtualH&E_semi-supervised_cycleGAN_26_08_26), so
the two pipelines can be run and compared side by side.

STATUS: Stage 1 (calibration), Stage 2 (rgb_to_he/he_to_rgb + validation),
and Stage 3 (preprocess_he, Dataset/DataLoader) are all in place. From this
point on, common.py has a hard dependency on torch (the Dataset class
below subclasses torch.utils.data.Dataset) -- earlier stages deliberately
avoided that so calibration work needed no torch install, but that's no
longer preserved past this point, since everything from here on is
training-side machinery that needs torch regardless.

Every stage script imports from this module and does its own setup
(CONFIG, discover_dataset(...), etc.) rather than relying on another
script's Interactive Window / kernel state -- same reasoning as the
existing project's common.py: VS Code gives each .py file its own
Interactive Window by default, so this keeps every stage runnable
independently, in any kernel.
"""

import functools
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from skimage.color import separate_stains, combine_stains
from skimage.transform import resize as sk_resize

Image.MAX_IMAGE_PIXELS = None  # H&E images are large; disable PIL's decompression-bomb guard


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET DISCOVERY & CORE FILTERING / CONFIDENCE WEIGHTING
# =============================================================================

@dataclass
class RuntimeInfo:
    manifest: Optional[pd.DataFrame] = None
    registration_summary: Optional[pd.DataFrame] = None
    included_core_ids: Optional[set] = None
    manifest_stats: Optional[dict] = None
    core_score_lookup: Optional[dict] = None


def discover_dataset(config: dict) -> RuntimeInfo:
    """Loads patch_manifest.csv + registration_summary.csv, applies
    core-level filtering, and prints a summary. See the data contract for
    the full column schema this expects.
    """
    manifest = pd.read_csv(config["PATCH_MANIFEST_CSV"])
    reg_summary = pd.read_csv(config["REGISTRATION_SUMMARY_CSV"])

    required_manifest_cols = {
        "core_id", "patch_id", "srs_x0", "srs_y0", "srs_x1", "srs_y1",
        "he_x0", "he_y0", "he_x1", "he_y1", "srs_path", "he_path",
        "valid", "split", "patch_nmi", "patch_edge_corr",
        "srs_tissue_fraction", "he_tissue_fraction",
    }
    missing = required_manifest_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"patch_manifest.csv is missing expected columns: {missing}")

    n_total = len(manifest)
    manifest = manifest[manifest["valid"]].copy()
    n_valid = len(manifest)
    print(f"Manifest: {n_total} candidate patches, {n_valid} valid.")

    # --- Core-level filtering ------------------------------------------
    reg_summary = reg_summary.set_index("core_id", drop=False)
    excluded_by_status = set(
        reg_summary.index[reg_summary["status"].isin(config["EXCLUDE_CORE_STATUSES"])]
    )
    excluded_by_score = set(
        reg_summary.index[reg_summary["score"] < config["MIN_CORE_SCORE"]]
    )
    all_cores = set(reg_summary.index)
    excluded_cores = excluded_by_status | excluded_by_score
    included_cores = all_cores - excluded_cores

    print(f"\nCores: {len(all_cores)} total.")
    if excluded_by_status:
        print(f"  Excluded by status {config['EXCLUDE_CORE_STATUSES']}: {sorted(excluded_by_status)}")
    if excluded_by_score:
        print(f"  Excluded by score < {config['MIN_CORE_SCORE']}: "
              f"{sorted((c, round(reg_summary.loc[c, 'score'], 3)) for c in excluded_by_score)}")
    print(f"  Included: {len(included_cores)} cores")

    manifest = manifest[manifest["core_id"].isin(included_cores)].copy()

    # --- Optional hard per-patch NMI filter -----------------------------
    if config["MIN_PATCH_NMI"] is not None:
        before = len(manifest)
        manifest = manifest[manifest["patch_nmi"] >= config["MIN_PATCH_NMI"]]
        print(f"\nMIN_PATCH_NMI={config['MIN_PATCH_NMI']}: dropped {before - len(manifest)} patches "
              f"({len(manifest)} remain).")

    # --- Thin-core warning (informational only) -------------------------
    counts = manifest.groupby("core_id").size()
    thin_cores = counts[counts < config["THIN_CORE_PATCH_WARNING_THRESHOLD"]]
    if len(thin_cores):
        print(f"\nWARNING: {len(thin_cores)} included core(s) have fewer than "
              f"{config['THIN_CORE_PATCH_WARNING_THRESHOLD']} valid patches after filtering:")
        for cid, n in thin_cores.items():
            print(f"    {cid}: {n} patches (split={reg_summary.loc[cid, 'split']})")

    # --- Split summary ----------------------------------------------------
    print("\nPatches per split (post-filtering):")
    print(manifest["split"].value_counts().to_string())
    print("\nCores per split (post-filtering):")
    print(manifest.drop_duplicates("core_id").groupby("split")["core_id"].count().to_string())

    # --- Precompute robust normalisation stats for confidence weighting ----
    nmi_p05, nmi_p95 = np.percentile(manifest["patch_nmi"], [5, 95])
    edge_p05, edge_p95 = np.percentile(manifest["patch_edge_corr"], [5, 95])
    manifest_stats = dict(nmi_p05=nmi_p05, nmi_p95=nmi_p95, edge_p05=edge_p05, edge_p95=edge_p95)

    # --- Core score lookup, normalised to [0, 1] across included cores ------
    included_scores = reg_summary.loc[list(included_cores), "score"]
    score_min, score_max = included_scores.min(), included_scores.max()
    score_range = max(score_max - score_min, 1e-8)
    core_score_lookup = {
        cid: float((reg_summary.loc[cid, "score"] - score_min) / score_range)
        for cid in included_cores
    }

    manifest = manifest.reset_index(drop=True)

    return RuntimeInfo(
        manifest=manifest,
        registration_summary=reg_summary,
        included_core_ids=included_cores,
        manifest_stats=manifest_stats,
        core_score_lookup=core_score_lookup,
    )


def compute_confidence_weight(row: pd.Series, config: dict, runtime: RuntimeInfo) -> float:
    """Combines patch-level (patch_nmi, patch_edge_corr, tissue fraction) and
    core-level (registration score) signals into one scalar in
    [CONF_WEIGHT_FLOOR, 1], used to scale morphology-loss terms per sample.
    Ported unchanged -- this has nothing to do with the color defect.
    """
    stats = runtime.manifest_stats

    def _norm01(value, lo, hi):
        if hi <= lo:
            return 0.5
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    nmi_norm = _norm01(row["patch_nmi"], stats["nmi_p05"], stats["nmi_p95"])
    edge_norm = _norm01(row["patch_edge_corr"], stats["edge_p05"], stats["edge_p95"])
    tissue = min(row["srs_tissue_fraction"], row["he_tissue_fraction"])
    core_score = runtime.core_score_lookup[row["core_id"]]

    raw = (
        config["CONF_W_NMI"] * nmi_norm
        + config["CONF_W_EDGE_CORR"] * edge_norm
        + config["CONF_W_TISSUE_FRACTION"] * tissue
        + config["CONF_W_CORE_SCORE"] * core_score
    )
    total_weight = (
        config["CONF_W_NMI"] + config["CONF_W_EDGE_CORR"]
        + config["CONF_W_TISSUE_FRACTION"] + config["CONF_W_CORE_SCORE"]
    )
    raw_norm = raw / total_weight

    floor = config["CONF_WEIGHT_FLOOR"]
    return float(floor + (1.0 - floor) * raw_norm)


def attach_confidence_weights(runtime: RuntimeInfo, config: dict) -> None:
    """Computes confidence_weight for every row and attaches it to
    runtime.manifest in place. Call once after discover_dataset().
    """
    runtime.manifest["confidence_weight"] = runtime.manifest.apply(
        lambda row: compute_confidence_weight(row, config, runtime), axis=1
    )


# =============================================================================
# H&E PATCH LOADING (ported unchanged -- pure I/O, not part of the color defect)
# =============================================================================

def _clip_bbox(x0, y0, x1, y1, width, height):
    """Round and clip a float bounding box to valid pixel indices.

    Bounding boxes from the manifest can fall slightly outside image bounds
    at core edges -- this is expected (see data contract), not a bug.
    """
    x0c = max(int(round(x0)), 0)
    y0c = max(int(round(y0)), 0)
    x1c = min(int(round(x1)), width)
    y1c = min(int(round(y1)), height)
    if x1c <= x0c or y1c <= y0c:
        raise ValueError(f"Degenerate crop after clipping: ({x0},{y0},{x1},{y1}) -> "
                          f"({x0c},{y0c},{x1c},{y1c}) for image size ({width}, {height})")
    return x0c, y0c, x1c, y1c


# Caches the full raw image per (path, flip) combination, per process -- see
# the existing project's common.py for the full reasoning. Same default.
_RAW_IMAGE_CACHE_SIZE = 8


@functools.lru_cache(maxsize=_RAW_IMAGE_CACHE_SIZE)
def _load_he_rgb_cached(path: str, flip_core_horizontal: bool) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    if flip_core_horizontal:
        img = img[:, ::-1, :]
    return img.copy()  # decouple from the cache -- callers must never get a mutable view into it


def load_he_patch_rgb(row: pd.Series, flip_core_horizontal: bool = False) -> np.ndarray:
    """Load and crop the original H&E RGB patch at native resolution.
    Returns uint8 array of shape (h, w, 3). The full image is cached per
    (path, flip) -- see _load_he_rgb_cached.
    """
    img = _load_he_rgb_cached(row["he_path"], flip_core_horizontal)
    height, width = img.shape[0], img.shape[1]
    x0, y0, x1, y1 = _clip_bbox(row["he_x0"], row["he_y0"], row["he_x1"], row["he_y1"], width, height)
    return img[y0:y1, x0:x1, :].copy()


# =============================================================================
# SRS PATCH LOADING (ported unchanged -- pure I/O/normalization, not part
# of the color defect this project is fixing)
# =============================================================================

@functools.lru_cache(maxsize=_RAW_IMAGE_CACHE_SIZE)
def _load_srs_cube_cached(path: str, flip_core_horizontal: bool) -> np.ndarray:
    cube = tifffile.imread(path)
    if cube.ndim != 3 or cube.shape[0] > cube.shape[1] or cube.shape[0] > cube.shape[2]:
        raise ValueError(
            f"Unexpected SRS tif shape {cube.shape} for {path}. "
            f"Expected channel-first (C, H, W) with a small C (e.g. 4). "
            f"If your files are channel-last (H, W, C), transpose before use."
        )
    if flip_core_horizontal:
        cube = cube[:, :, ::-1]
    return cube.copy()  # decouple from the cache -- callers must never get a mutable view into it


def load_srs_patch(row: pd.Series, channel_indices: list, flip_core_horizontal: bool = False) -> np.ndarray:
    """Load and crop the requested channels from the original SRS tif.
    Returns float32 array of shape (len(channel_indices), h, w) at native
    resolution (not yet resized). The full image is cached per (path, flip)
    -- see _load_srs_cube_cached -- so repeated calls for different patches
    from the same core don't re-read the file from disk each time.

    flip_core_horizontal: if True, the WHOLE loaded image is mirrored
    left-right (flipped around the vertical axis) BEFORE cropping with the
    manifest's bounding box -- this matters if the manifest's box
    coordinates were computed in a coordinate frame that assumes this
    whole-image flip (e.g. because the upstream registration step flipped
    the core before registering it).
    """
    cube = _load_srs_cube_cached(row["srs_path"], flip_core_horizontal)
    n_channels, height, width = cube.shape
    if max(channel_indices) >= n_channels:
        raise ValueError(
            f"SRS_CHANNEL_INDICES {channel_indices} out of range for a "
            f"{n_channels}-channel cube at {row['srs_path']}."
        )
    x0, y0, x1, y1 = _clip_bbox(row["srs_x0"], row["srs_y0"], row["srs_x1"], row["srs_y1"], width, height)
    patch = cube[channel_indices, y0:y1, x0:x1]
    return patch.astype(np.float32)


def resize_channels_first(array: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a (C, h, w) array to (C, target_size, target_size), per-channel."""
    c = array.shape[0]
    out = np.empty((c, target_size, target_size), dtype=np.float32)
    for i in range(c):
        out[i] = sk_resize(
            array[i], (target_size, target_size),
            order=1, preserve_range=True, anti_aliasing=True,
        )
    return out


def percentile_normalize(array: np.ndarray, percentiles: tuple) -> np.ndarray:
    """Per-channel percentile clip then rescale to [-1, 1]. array is (C, H, W).
    Used for SRS input (which has no calibration/color-defect issue) --
    NOT used for H&E anymore, since that's rgb_to_he's job now.
    """
    out = np.empty_like(array, dtype=np.float32)
    lo_p, hi_p = percentiles
    for i in range(array.shape[0]):
        lo, hi = np.percentile(array[i], [lo_p, hi_p])
        if hi <= lo:
            out[i] = 0.0
            continue
        chan = np.clip(array[i], lo, hi)
        out[i] = 2.0 * (chan - lo) / (hi - lo) - 1.0
    return out


def preprocess_srs(row: pd.Series, config: dict) -> np.ndarray:
    """Full SRS preprocessing pipeline: load -> select channels -> resize ->
    normalise. Returns float32 (C, NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE)
    in [-1, 1]. Unchanged from the existing project -- SRS has no
    calibration defect to fix.
    """
    raw = load_srs_patch(row, config["SRS_CHANNEL_INDICES"],
                          flip_core_horizontal=config.get("SRS_CORE_FLIP_HORIZONTAL", False))
    resized = resize_channels_first(raw, config["NETWORK_INPUT_SIZE"])
    normed = percentile_normalize(resized, config["SRS_NORM_PERCENTILES"])
    return normed


def preprocess_he(row: pd.Series, config: dict, calibration: "StainCalibration") -> np.ndarray:
    """Full H&E preprocessing pipeline: load RGB -> resize RGB -> rgb_to_he.
    Returns float32 (2, NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE) in [-1, 1].

    This is what REPLACES the old project's preprocess_he: deconvolution
    now uses the calibrated matrix and normalization uses the global
    percentile range, both from `calibration`, instead of the generic
    rgb2hed matrix and a per-patch percentile clip. Deconvolution still
    happens AFTER resizing (it's a per-pixel linear operation, so this is
    far cheaper than deconvolving at native resolution) -- same reasoning
    as the old pipeline, just calling rgb_to_he instead of the old
    he_rgb_to_hed_channels + percentile_normalize combination.
    """
    rgb = load_he_patch_rgb(row, flip_core_horizontal=config.get("HE_CORE_FLIP_HORIZONTAL", False))
    rgb_resized = sk_resize(
        rgb, (config["NETWORK_INPUT_SIZE"], config["NETWORK_INPUT_SIZE"]),
        order=1, preserve_range=True, anti_aliasing=True,
    ).astype(np.uint8)
    return rgb_to_he(rgb_resized, calibration)


# =============================================================================
# CALIBRATED COLOR TRANSFORM (Stage 2 -- built from Stage 1's
# stain_calibration.npz)
# =============================================================================
# This replaces BOTH defects the whole rebuild exists to fix:
#   - the generic Ruifrok & Johnston stain matrix (skimage's rgb2hed default)
#     is replaced by the calibrated matrix fit in 01_data_pipeline.py
#   - per-patch percentile normalization is replaced by the global H/E
#     ranges from the same calibration run
#
# rgb_to_he / he_to_rgb are NOT yet wired into any preprocessing, training,
# or visualization code. Per the project's agreed validation order, that
# only happens after these two functions are round-trip tested on real
# H,E channels -- see 01_data_pipeline.py's Stage 2 cells for that test.

@dataclass
class StainCalibration:
    rgb_from_hed: np.ndarray    # (3, 3) -- stain color vectors, HED -> RGB
    hed_from_rgb: np.ndarray    # (3, 3) -- inverse, RGB -> HED (= inv(rgb_from_hed))
    global_h_range: tuple       # (lo, hi) raw calibrated-OD range for Hematoxylin
    global_e_range: tuple       # (lo, hi) raw calibrated-OD range for Eosin
    nuclear_h_threshold: float  # carried through for convenience/provenance;
                                 # NOT used by rgb_to_he/he_to_rgb themselves --
                                 # that's the nuclear-density loss's business


def load_stain_calibration(path: str) -> StainCalibration:
    """Loads the .npz written by 01_data_pipeline.py's Stage 1d."""
    data = np.load(path)
    return StainCalibration(
        rgb_from_hed=data["rgb_from_hed"],
        hed_from_rgb=data["hed_from_rgb"],
        global_h_range=tuple(data["global_h_range"]),
        global_e_range=tuple(data["global_e_range"]),
        nuclear_h_threshold=float(data["nuclear_h_threshold"]),
    )


def _clip_rescale(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """[lo, hi] -> [-1, 1], clipping first. Same convention as the existing
    project's percentile_normalize, just against a fixed global range
    instead of a per-array one.
    """
    if hi <= lo:
        return np.zeros_like(value)
    clipped = np.clip(value, lo, hi)
    return 2.0 * (clipped - lo) / (hi - lo) - 1.0


def _unclip_rescale(value_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse of _clip_rescale: [-1, 1] -> [lo, hi]. NOTE this is only a
    true inverse for values that started out inside [-1, 1] -- clipping in
    the forward direction is lossy, so a value that got clamped can't be
    recovered exactly. That's fine for our actual use cases: real data run
    through rgb_to_he is already in [-1, 1] by construction, and a
    generator's tanh output layer guarantees the same for fake data.
    """
    return (value_norm + 1.0) / 2.0 * (hi - lo) + lo


def rgb_to_he(rgb_uint8: np.ndarray, calibration: StainCalibration) -> np.ndarray:
    """Converts an (h, w, 3) uint8 RGB H&E patch to a (2, h, w) float32
    array of Hematoxylin, Eosin channels, deconvolved with the CALIBRATED
    matrix and normalized against the GLOBAL percentile range -- both
    pieces from Stage 1, both in [-1, 1] on output.
    """
    hed = separate_stains(rgb_uint8, calibration.hed_from_rgb)
    h_raw, e_raw = hed[..., 0], hed[..., 1]
    h_norm = _clip_rescale(h_raw, *calibration.global_h_range)
    e_norm = _clip_rescale(e_raw, *calibration.global_e_range)
    return np.stack([h_norm, e_norm], axis=0).astype(np.float32)


def he_to_rgb(h_norm: np.ndarray, e_norm: np.ndarray, calibration: StainCalibration) -> np.ndarray:
    """Inverse of rgb_to_he: (h, w) H and E channels in [-1, 1] -- real or
    generator-produced -- back to (h, w, 3) uint8 RGB.

    The residual ("DAB slot") third channel that separate_stains produces
    (and that rgb_to_he discards) is ZEROED here before inverting, per the
    documented decision in Stage 1's calibration script: it has no
    physical meaning for H&E, the generator was never trained to produce
    it, and zeroing is the standard convention for this. If a Stage 2
    round-trip test shows this approximation is worse than expected, that
    decision is the first thing to revisit -- not a hidden assumption
    buried in this function.

    Expects plain numpy arrays. If h_norm/e_norm come from a torch
    generator's output, convert with .detach().cpu().numpy() first.
    """
    h_raw = _unclip_rescale(h_norm, *calibration.global_h_range)
    e_raw = _unclip_rescale(e_norm, *calibration.global_e_range)
    residual = np.zeros_like(h_raw)
    hed = np.stack([h_raw, e_raw, residual], axis=-1)
    rgb_float = combine_stains(hed, calibration.rgb_from_hed)
    return np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)


# =============================================================================
# DATASET / DATALOADER (Stage 3 -- ported from the existing project, with
# one deliberate change: the Dataset now needs a StainCalibration to pass
# into preprocess_he, since that's no longer parameter-free)
# =============================================================================

class QuasiPairedPatchDataset(Dataset):
    """Sampling strategy depends on config['MODEL_MODE']:
      - "baseline": SRS and H&E come from two INDEPENDENT random patches
        (patch_id ignored), matching classic unpaired CycleGAN.
      - "morphology" / "morphology_nuclear": SRS and H&E come from the SAME
        patch_id (quasi-paired, tissue-level correspondence only -- never
        treat this as pixel-aligned ground truth).

    Requires a StainCalibration (see load_stain_calibration) -- loaded ONCE
    by the caller and passed in here, not reloaded per __getitem__ call.
    """

    def __init__(self, manifest: pd.DataFrame, config: dict, split: str, calibration: StainCalibration):
        self.config = config
        self.calibration = calibration
        self.mode = config["MODEL_MODE"]
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No patches found for split='{split}'. Check filtering upstream.")

    def __len__(self):
        return len(self.df)

    def _random_unpaired_he_row(self) -> pd.Series:
        """Independent random draw for 'baseline' mode -- see the existing
        project's common.py for the full reasoning on why this is a plain
        independent draw rather than a stateful per-epoch counter (short
        version: DataLoader workers each get their own copy of Dataset
        state under NUM_WORKERS > 0, so a per-epoch counter silently stops
        meaning what it looks like it means).
        """
        idx = np.random.randint(len(self.df))
        return self.df.iloc[idx]

    def __getitem__(self, idx):
        srs_row = self.df.iloc[idx]

        if self.mode == "baseline":
            he_row = self._random_unpaired_he_row()
        elif self.mode in ("morphology", "morphology_nuclear"):
            he_row = srs_row  # same patch_id -> quasi-paired, tissue-level only
        else:
            raise ValueError(f"Unknown MODEL_MODE: {self.mode}")

        srs_tensor = torch.from_numpy(preprocess_srs(srs_row, self.config))
        he_tensor = torch.from_numpy(preprocess_he(he_row, self.config, self.calibration))

        return dict(
            srs=srs_tensor,
            he=he_tensor,
            confidence_weight=torch.tensor(float(he_row["confidence_weight"]), dtype=torch.float32),
            srs_core_id=srs_row["core_id"],
            srs_patch_id=srs_row["patch_id"],
            he_core_id=he_row["core_id"],
            he_patch_id=he_row["patch_id"],
            split=srs_row["split"],
        )


def _dataloader_worker_init_fn(worker_id: int):
    """Without this, numpy's and Python's global RNGs are simply forked from
    the parent process when DataLoader spawns workers -- meaning every
    worker can end up drawing the SAME sequence of "random" numbers (a
    well-known PyTorch gotcha; torch's own RNG is reseeded per worker
    automatically, numpy's and random's are not). This reseeds both
    explicitly, per worker, from torch's already-correct per-worker seed.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(runtime: RuntimeInfo, config: dict, split: str, shuffle: bool,
                      calibration: StainCalibration, required: bool = True):
    """Returns (loader, dataset), or (None, None) if the split is empty and
    `required` is False. The train split is always required.
    """
    n_available = (runtime.manifest["split"] == split).sum()
    if n_available == 0:
        msg = f"No valid patches for split='{split}' after filtering."
        if required:
            raise ValueError(msg + " This split is required -- check MIN_CORE_SCORE "
                              "and core status filtering above.")
        else:
            print(f"WARNING: {msg} Skipping this split for now.")
            return None, None

    dataset = QuasiPairedPatchDataset(runtime.manifest, config, split=split, calibration=calibration)
    num_workers = config["NUM_WORKERS"]
    loader = DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_dataloader_worker_init_fn if num_workers > 0 else None,
        # Keeps worker processes alive across epochs instead of respawning
        # them every epoch -- matters a lot on Windows, where process spawn
        # (the only method available there) is far more expensive than
        # Linux's fork.
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset


def random_rot90_pair(real_S: torch.Tensor, real_H: torch.Tensor):
    """Discriminator-memorization mitigation: applies an independent random
    90-degree-multiple rotation (0/90/180/270) to each sample in the batch,
    identically to its SRS and H&E patch so spatial correspondence is
    preserved for the macro/nuclear losses (which compare fake_H against
    real_H pixel-for-pixel and require the pair to stay registered).

    Call this ONCE, immediately after real_S/real_H are loaded onto device
    and BEFORE anything else touches them (generator, cycle, identity,
    macro/nuclear, discriminators) -- everything downstream is computed
    from these two tensors, so this single call site is sufficient.

    Must be called identically (same call, same position in the training
    loop) in 02_train_baseline_cyclegan.py, 04_train_morphology_cyclegan.py,
    and 05_train_morphology_nuclear_cyclegan.py -- adding it to only one of
    the three re-opens the same kind of uncontrolled-variable problem as the
    LEARNING_RATE_D mismatch between the baseline/morphology scripts and the
    morphology_nuclear script (see project notes). Do NOT use this for the
    val/test loaders -- evaluation should see patches in their original,
    un-augmented orientation.

    Returns new tensors (does not modify real_S/real_H in place), since
    both may still be referenced elsewhere by the caller before this point
    in some scripts.
    """
    aug_S = real_S.clone()
    aug_H = real_H.clone()
    for i in range(real_S.shape[0]):
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            aug_S[i] = torch.rot90(real_S[i], k, dims=[-2, -1])
            aug_H[i] = torch.rot90(real_H[i], k, dims=[-2, -1])
    return aug_S, aug_H
