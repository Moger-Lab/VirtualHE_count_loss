"""
Shared machinery for the count-based nuclear loss project
(VirtualH&E_count_loss).

Everything here is NEW code. It imports v1's common.py / models.py read-only
(from the RTX4500 project) for the stain calibration, rgb_to_he / he_to_rgb
and the network architectures, and never modifies them. It reads v2's
registration outputs read-only.

Data source: the v2 coregistration. The H&E is already warped onto the SRS
pixel grid, so one (y0, x0, 441) slice indexes both modalities at the same
tissue. NO flip is applied anywhere on this path: v2 encodes the SRS/H&E
mirror inside the warped H&E. Applying v1's SRS_CORE_FLIP_HORIZONTAL on top
would double-flip (see DECISION_LOG.md, D-01).
"""

import functools
import os
import sys

import numpy as np
import pandas as pd

_this_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# PATHS
# =============================================================================
import vhe_paths  # repository: all machine-specific locations live in vhe_paths.py
PY_SCRIPTS = os.path.dirname(_this_dir)
V1_DIR = os.path.join(_this_dir, "vhe_v1")  # bundled copy of the helper modules (common.py, models.py, calibration)
V1_NONAUG_DIR = V1_DIR  # only used by historical diagnostics not part of the paper
V2_DIR = vhe_paths.V2_DIR
V2_RESULTS = os.path.join(V2_DIR, "results")
ALIGNED_DIR = vhe_paths.ALIGNED
WORK_DIR = vhe_paths.WORK

# The single stain calibration used by v1 Model 3 (identical bytes in the
# non-augmented and RTX4500 folders). Kept so that A0 is "Model 3 as-is".
CALIBRATION_PATH = os.path.join(V1_DIR, "calibration", "stain_calibration.npz")

# Old (v1) manifest with locally valid paths, used only for the Model 3
# diagnostic and the loader-equivalence test.
OLD_MANIFEST_CSV = os.path.join(V1_NONAUG_DIR, "Paired Images", "patch_manifest.csv")
OLD_REGSUM_CSV = os.path.join(V1_NONAUG_DIR, "Paired Images", "registration_summary.csv")

PATCH_NATIVE = 441
RENDER = 512

# v1 evaluation protocol (07_evaluate_models.py / run_phase2_stardist.py)
STARDIST_MODEL_NAME = "2D_versatile_he"
STARDIST_NORM_PERCENTILES = (1.0, 99.8)
MIN_NUCLEUS_AREA_PX = 10

# v1 FINDINGS density tercile cutoffs (nuclei per 100 um patch)
BUCKET_EDGES = (33, 56)
BUCKET_NAMES = ("low", "med", "high")

LEGACY_TEST = ("B-8", "C-2", "C-3", "C-4", "E-7", "F-4", "F-6", "F-8", "G-1", "G-4", "H-6")
UNPAIRED_ONLY = ("A-5", "A-7", "A-9", "B-9", "G-2")


def import_v1():
    """Imports v1's common and models modules from the RTX4500 folder,
    read-only. Returns (common, models)."""
    if V1_DIR not in sys.path:
        sys.path.insert(0, V1_DIR)
    try:
        import common as v1_common
        import models as v1_models
    except ModuleNotFoundError as e:
        if e.name not in ("common", "models"):
            raise
        raise ModuleNotFoundError(
            f"Could not import v1 common.py / models.py from {V1_DIR}. "
            "This project reuses them read-only; check the folder exists."
        ) from e
    return v1_common, v1_models


def density_bucket(count):
    count = np.asarray(count)
    return np.where(count < BUCKET_EDGES[0], "low", np.where(count < BUCKET_EDGES[1], "med", "high"))


# =============================================================================
# V2 DATA ACCESS
# =============================================================================

def read_csv_bom(path: str) -> pd.DataFrame:
    """v2's CSVs may carry a UTF-8 BOM (DECISION_LOG handover item 2)."""
    return pd.read_csv(path, encoding="utf-8-sig")


def load_v2_tables():
    """Returns (manifest, cv_design, transforms). Excluded cores A-1 and A-6
    appear in none of these files."""
    manifest = read_csv_bom(os.path.join(V2_RESULTS, "15_patch_manifest.csv"))
    cv = read_csv_bom(os.path.join(V2_RESULTS, "14_cv_design.csv"))
    tr = read_csv_bom(os.path.join(V2_RESULTS, "16_transforms.csv"))
    manifest["fold"] = manifest["fold"].astype("Int64")
    cv["fold"] = cv["fold"].astype("Int64")
    manifest["patch_uid"] = (manifest["core"] + "_" + manifest["role"] + "_"
                             + manifest["y0"].astype(str) + "_" + manifest["x0"].astype(str))
    return manifest, cv, tr


def split_of(manifest: pd.DataFrame, val_fold: int = 0) -> pd.Series:
    """Per-row split label under the screening design:
      rotating & fold == val_fold  -> 'val'
      rotating & fold != val_fold  -> 'train'
      legacy_test                  -> 'legacy_test'  (never used for selection)
      unpaired_only                -> 'unpaired'     (adversarial/cycle only)
    """
    out = pd.Series("train", index=manifest.index, dtype=object)
    rot = manifest["group"] == "rotating"
    out[rot & (manifest["fold"] == val_fold)] = "val"
    out[manifest["group"] == "legacy_test"] = "legacy_test"
    out[manifest["group"] == "unpaired_only"] = "unpaired"
    return out


_ARRAY_CACHE = {}


def core_arrays(core: str):
    """(srs, he) memory-mapped arrays for a core, one handle per core per
    process. srs: (4, 4408, 4408) uint16, native, UNFLIPPED. he: (4408, 4408,
    3) uint8, warped onto the SRS grid (mirror already applied)."""
    if core not in _ARRAY_CACHE:
        sp = os.path.join(ALIGNED_DIR, f"{core}_srs.npy")
        hp = os.path.join(ALIGNED_DIR, f"{core}_he.npy")
        if not (os.path.exists(sp) and os.path.exists(hp)):
            raise FileNotFoundError(
                f"Aligned arrays for {core} not found in {ALIGNED_DIR}. Regenerate with "
                r"Python_Scripts\VirtualH&E_v2_transfer\regenerate_aligned.py (about 2 min)."
            )
        _ARRAY_CACHE[core] = (np.load(sp, mmap_mode="r"), np.load(hp, mmap_mode="r"))
    return _ARRAY_CACHE[core]


def v2_patch_native(core: str, y0: int, x0: int, size: int = PATCH_NATIVE, channels=None):
    """Returns (srs (C, size, size) float32, he (size, size, 3) uint8)."""
    srs, he = core_arrays(core)
    ch = list(range(srs.shape[0])) if channels is None else list(channels)
    s = np.asarray(srs[ch, y0:y0 + size, x0:x0 + size]).astype(np.float32)
    h = np.asarray(he[y0:y0 + size, x0:x0 + size])
    return s, h


def resize_rgb(rgb: np.ndarray, size: int = RENDER) -> np.ndarray:
    """Same anti-aliased bilinear resize as v1's preprocess_he."""
    from skimage.transform import resize as sk_resize
    return sk_resize(rgb, (size, size), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)


def resize_chw(arr: np.ndarray, size: int = RENDER) -> np.ndarray:
    from skimage.transform import resize as sk_resize
    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = sk_resize(arr[i], (size, size), order=1, preserve_range=True, anti_aliasing=True)
    return out


# =============================================================================
# STARDIST (v1 protocol)
# =============================================================================

# =============================================================================
# SRS NORMALISATION (flag SRS_NORM_MODE = "per_patch" | "per_core")
# =============================================================================
SRS_NORM_CACHE = os.path.join(WORK_DIR, "srs_norm", "per_core_percentiles.csv")
PER_CORE_PERCENTILES = (1.0, 99.5)


def srs_tissue_mask(srs_full, stride: int = 4) -> np.ndarray:
    """SRS-only tissue mask at `stride` (independent of the H&E and of
    registration, so it is valid for unpaired cores too). CARS (ch 2) is
    generated throughout tissue. Background level and spread come from the
    frame corners, which lie outside the inscribed disc and are background
    by construction (v2 DECISION_LOG stage 5)."""
    from scipy.ndimage import gaussian_filter, binary_opening
    cars = gaussian_filter(np.asarray(srs_full[2, ::stride, ::stride]).astype(np.float32), 2)
    h, w = cars.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    corner = r > 0.5 * min(h, w)
    bg = cars[corner]
    med = np.median(bg); mad = 1.4826 * np.median(np.abs(bg - med)) + 1e-6
    mask = cars > med + 6 * mad
    return binary_opening(mask, iterations=2)


def compute_per_core_percentiles(cores, out_path: str = SRS_NORM_CACHE, stride: int = 4) -> pd.DataFrame:
    """p1 / p99.5 per channel over tissue pixels of the full cube (sampled at
    `stride`), one row per core. Cached to disk; returns the table."""
    recs = []
    for core in cores:
        srs, _ = core_arrays(core)
        mask = srs_tissue_mask(srs, stride)
        rec = dict(core=core, tissue_frac=float(mask.mean()))
        for c in range(srs.shape[0]):
            v = np.asarray(srs[c, ::stride, ::stride])[mask].astype(np.float32)
            lo, hi = np.percentile(v, PER_CORE_PERCENTILES)
            rec[f"ch{c}_lo"] = float(lo); rec[f"ch{c}_hi"] = float(hi)
        recs.append(rec)
        print(f"  {core}: tissue {rec['tissue_frac']:.2f}  "
              + "  ".join(f"ch{c} [{rec[f'ch{c}_lo']:.0f}, {rec[f'ch{c}_hi']:.0f}]" for c in range(4)), flush=True)
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


CARS_THR_CACHE = os.path.join(WORK_DIR, "srs_norm", "cars_tissue_threshold.csv")


EMPTY_K_MAD = 3.0


def compute_cars_thresholds(cores, stride: int = 4) -> pd.DataFrame:
    """Per-core background thresholds for BOTH CARS (ch 2) and SRS (ch 1):
    corner median + EMPTY_K_MAD x MAD, after Gaussian smoothing sigma = 2 at
    `stride` (8 native px). A pixel is SRS-empty only if it is at background
    level in both channels (A6, D-18; CARS alone mislabels dim stroma, B-8)."""
    from scipy.ndimage import gaussian_filter
    recs = []
    for core in cores:
        srs, _ = core_arrays(core)
        rec = dict(core=core)
        for ch, name in ((2, "cars"), (1, "srs")):
            img = gaussian_filter(np.asarray(srs[ch, ::stride, ::stride]).astype(np.float32), 2)
            h, w = img.shape
            yy, xx = np.mgrid[0:h, 0:w]
            corner = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) > 0.5 * min(h, w)
            bg = img[corner]; med = np.median(bg); mad = 1.4826 * np.median(np.abs(bg - med)) + 1e-6
            rec[f"{name}_thr"] = float(med + EMPTY_K_MAD * mad)
        recs.append(rec)
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(CARS_THR_CACHE), exist_ok=True)
    df.to_csv(CARS_THR_CACHE, index=False)
    return df


@functools.lru_cache(maxsize=1)
def cars_thresholds() -> dict:
    if not os.path.exists(CARS_THR_CACHE):
        raise FileNotFoundError(f"{CARS_THR_CACHE} missing: run s11_cars_thresholds.py first.")
    d = pd.read_csv(CARS_THR_CACHE)
    return {r.core: (r.cars_thr, r.srs_thr) for r in d.itertuples()}


def srs_empty_mask(core: str, srs_native: np.ndarray, size: int = RENDER, dilate_px: int = 12,
                   min_open_px: int = 20) -> np.ndarray:
    """1 where the SRS shows NO tissue in this patch, at the 512 grid.
    srs_native: (2, h, w) = [SRS ch 1, CARS ch 2]. Empty = both channels,
    smoothed (sigma 8 native px), at background level; tissue dilated by
    ~2.5 um so edges are never penalised; empty regions opened with a
    ~4.5 um element so only lumen-sized gaps count."""
    from scipy.ndimage import gaussian_filter, binary_dilation, binary_opening
    from skimage.transform import resize as sk_resize
    if srs_native.ndim != 3 or srs_native.shape[0] != 2:
        raise ValueError(f"srs_empty_mask expects (2, h, w) = [SRS ch1, CARS ch2], got {srs_native.shape}")
    cars_thr, srs_thr = cars_thresholds()[core]
    s1 = gaussian_filter(srs_native[0].astype(np.float32), 8)
    s2 = gaussian_filter(srs_native[1].astype(np.float32), 8)
    tissue = binary_dilation((s1 > srs_thr) | (s2 > cars_thr), iterations=dilate_px)
    empty = binary_opening(~tissue, iterations=min_open_px)
    return (sk_resize(empty.astype(np.float32), (size, size), order=0, anti_aliasing=False) > 0.5).astype(np.float32)


@functools.lru_cache(maxsize=1)
def per_core_percentile_table(path: str = SRS_NORM_CACHE) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run s02b_srs_norm_cache.py first (SRS_NORM_MODE='per_core').")
    df = pd.read_csv(path)
    return {r.core: r for r in df.itertuples(index=False)}


TYPE_CHANNELS = (0, 3)  # SHG, TPEF: report tissue TYPE (v2 DECISION_LOG, channel semantics)


def normalize_srs(srs512: np.ndarray, core: str, channels, mode: str, type_channel_norm: str = "global") -> np.ndarray:
    """srs512: (C, H, W) float32 raw counts for `channels`. Returns [-1, 1].
    per_patch = v1's percentile_normalize (1, 99) on this patch alone.
    per_core  = fixed per-core, per-channel p1 / p99.5 over tissue, EXCEPT
                the tissue-type channels (SHG 0, TPEF 3) when
                type_channel_norm == 'global': those use one cohort range
                pooled over training cores, so a core with no collagen keeps
                a near-empty SHG channel instead of having its noise
                stretched to full range (DECISION_LOG D-06)."""
    out = np.empty_like(srs512, dtype=np.float32)
    if mode == "per_patch":
        for i in range(srs512.shape[0]):
            lo, hi = np.percentile(srs512[i], [1.0, 99.0])
            out[i] = 0.0 if hi <= lo else 2.0 * (np.clip(srs512[i], lo, hi) - lo) / (hi - lo) - 1.0
        return out
    if mode == "per_core":
        table = per_core_percentile_table()
        for i, c in enumerate(channels):
            if type_channel_norm not in ("global", "per_core"):
                raise ValueError(f"TYPE_CHANNEL_NORM must be 'global' or 'per_core', got {type_channel_norm!r}")
            use_global = (c in TYPE_CHANNELS) and type_channel_norm == "global"
            row = table["__GLOBAL__"] if use_global else table[core]
            lo, hi = getattr(row, f"ch{c}_lo"), getattr(row, f"ch{c}_hi")
            out[i] = 2.0 * (np.clip(srs512[i], lo, hi) - lo) / max(hi - lo, 1e-6) - 1.0
        return out
    raise ValueError(f"Unknown SRS_NORM_MODE {mode!r}; use 'per_patch' or 'per_core'.")


# =============================================================================
# STARDIST CACHE ACCESS, TARGET HEATMAPS, PER-CELL COUNTS
# =============================================================================
STARDIST_CACHE = os.path.join(WORK_DIR, "stardist_v2", "stardist_cache.pkl")


@functools.lru_cache(maxsize=1)
def stardist_cache(path: str = STARDIST_CACHE) -> dict:
    import pickle
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run s02a_stardist_cache.py (shards, then --merge).")
    with open(path, "rb") as f:
        return pickle.load(f)


def heatmap_from_centroids(cy, cx, sigma: float, size: int = RENDER) -> np.ndarray:
    """One unit-integral Gaussian per centroid. Each Gaussian is renormalised
    to its in-image mass, so the map integrates to the count exactly even
    for nuclei at the border. Sub-pixel centroids are bilinear-splatted."""
    from scipy.ndimage import gaussian_filter
    from scipy.special import erf
    m = np.zeros((size, size), np.float64)
    if len(cy) == 0:
        return m.astype(np.float32)
    cy = np.clip(np.asarray(cy, np.float64), 0, size - 1); cx = np.clip(np.asarray(cx, np.float64), 0, size - 1)
    s2 = sigma * np.sqrt(2)
    # in-image mass of a Gaussian centred at c over pixel range [-0.5, size-0.5]
    my = 0.5 * (erf((size - 0.5 - cy) / s2) - erf((-0.5 - cy) / s2))
    mx = 0.5 * (erf((size - 0.5 - cx) / s2) - erf((-0.5 - cx) / s2))
    w = 1.0 / np.clip(my * mx, 1e-3, None)
    y0 = np.floor(cy).astype(int); x0 = np.floor(cx).astype(int)
    fy = cy - y0; fx = cx - x0
    y1 = np.minimum(y0 + 1, size - 1); x1 = np.minimum(x0 + 1, size - 1)
    np.add.at(m, (y0, x0), w * (1 - fy) * (1 - fx))
    np.add.at(m, (y0, x1), w * (1 - fy) * fx)
    np.add.at(m, (y1, x0), w * fy * (1 - fx))
    np.add.at(m, (y1, x1), w * fy * fx)
    out = gaussian_filter(m, sigma, mode="constant", truncate=4.0)
    # renormalisation above is analytic; correct the small discretisation
    # residual so the integral equals the count to float precision
    tot = out.sum()
    if tot > 0:
        out *= len(cy) / tot
    return out.astype(np.float32)


def cell_counts_from_centroids(cy, cx, grid: int, size: int = RENDER) -> np.ndarray:
    cell = size / grid
    gy = np.clip((np.asarray(cy) // cell).astype(int), 0, grid - 1)
    gx = np.clip((np.asarray(cx) // cell).astype(int), 0, grid - 1)
    c = np.zeros((grid, grid), np.float32)
    np.add.at(c, (gy, gx), 1.0)
    return c


def median_equiv_radius(uids) -> float:
    """Median StarDist equivalent radius (512 grid px) over the given patches."""
    sc = stardist_cache()
    areas = np.concatenate([sc[u][2] for u in uids if len(sc[u][2])])
    return float(np.sqrt(np.median(areas) / np.pi))


def q95_cell_counts(uids, grid: int) -> float:
    """95th percentile of per-cell real StarDist counts over training patches."""
    sc = stardist_cache()
    cells = np.concatenate([cell_counts_from_centroids(sc[u][0], sc[u][1], grid).ravel() for u in uids])
    return float(np.percentile(cells, 95))


@functools.lru_cache(maxsize=1)
def stardist_model():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from stardist.models import StarDist2D
    return StarDist2D.from_pretrained(STARDIST_MODEL_NAME)


def detect_nuclei(rgb_uint8: np.ndarray):
    """v1 protocol. Returns (cy, cx, area) float arrays for nuclei with
    area >= MIN_NUCLEUS_AREA_PX, plus the label image."""
    from csbdeep.utils import normalize as sd_normalize
    from skimage.measure import regionprops
    pmin, pmax = STARDIST_NORM_PERCENTILES
    labels, _ = stardist_model().predict_instances(sd_normalize(rgb_uint8, pmin, pmax), show_tile_progress=False)
    cy, cx, ar = [], [], []
    for p in regionprops(labels):
        if p.area >= MIN_NUCLEUS_AREA_PX:
            cy.append(p.centroid[0]); cx.append(p.centroid[1]); ar.append(p.area)
    return (np.array(cy, np.float32), np.array(cx, np.float32), np.array(ar, np.float32)), labels
