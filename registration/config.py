"""Central paths and cohort constants for the v2 virtual-H&E build.

Every path is local. Nothing in this project reads from or writes to a remote
location (HANDOVER.md section 8).

Path note: on the original host the profile folder name differed from the user name; always
resolve from USERPROFILE, never from USERNAME.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- host anchors -----------------------------------------------------------

USERPROFILE = Path(os.environ.get("USERPROFILE", str(Path.home())))
# repository: the original host check is disabled; set the VHE_* variables instead

ONEDRIVE = USERPROFILE / "OneDrive - University of Exeter"
PYTHON_SCRIPTS = ONEDRIVE / "Python_Scripts"

# --- source data (read-only) ------------------------------------------------

DATA_ROOT = Path(os.environ.get("VHE_RAW_DATA", Path(__file__).resolve().parent.parent / "data" / "raw"))
SRS_DIR = DATA_ROOT / "fsSRS"
HE_DIR = DATA_ROOT / "H&E"

# Filename conventions as they exist on disk.
#
# SRS is uniform: `<core>_fSRS.tif`.
#
# H&E is NOT uniform, and HANDOVER.md documents only one of the two forms it
# takes. Row A (9 files) is bare `<core>.png`; rows B-H (69 files) carry the
# documented `<core>_H&E.png`. Parsing on either convention alone silently
# drops the other set - assuming the documented form alone loses all 9 row-A
# cores and reduces the paired cohort from 77 to 69.
SRS_SUFFIX = "_fSRS.tif"
HE_GLOB = "*.png"
HE_OPTIONAL_SUFFIX = "_H&E"


def he_core_id(path) -> str:
    """Core ID from an H&E filename, tolerating both naming conventions."""
    stem = path.stem
    if stem.endswith(HE_OPTIONAL_SUFFIX):
        stem = stem[: -len(HE_OPTIONAL_SUFFIX)]
    return stem

# --- prior work (read-only, for comparison only; no code is imported) --------

V1_CYCLEGAN = PYTHON_SCRIPTS / "VirtualH&E_semi-supervised_cycleGAN_30_08_26"
V1_ANALYSIS = PYTHON_SCRIPTS / "VirtualH&E_semi-supervised_Analysis_v1"
V1_FINDINGS = V1_ANALYSIS / "density_correction_screening" / "FINDINGS_for_publication.md"

# --- this project -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Heavy artefacts (patch caches, checkpoints, intermediate arrays) are written to a work
# folder outside the repository; set VHE_V2_WORK to choose it.
WORK_ROOT = Path(os.environ.get("VHE_V2_WORK", os.path.join(os.path.expanduser("~"), "vhe_v2_work")))
PATCH_DIR = WORK_ROOT / "patches"
CHECKPOINT_DIR = WORK_ROOT / "checkpoints"
CACHE_DIR = WORK_ROOT / "cache"

# --- SRS channel semantics (confirmed by the project owner) -----------------
# HANDOVER.md section 2. Channel 1 is a BROADBAND AVERAGE over 2700-3100 cm-1:
# lipid and protein summed, so no CH3-CH2 difference is recoverable.
SRS_CHANNELS = {
    0: "SHG",           # collagen / fibrillar stroma
    1: "fsSRS_broad",   # broadband 2700-3100 cm-1, lipid + protein summed
    2: "CARS",          # similar chemistry, different process + background
    3: "TPF",           # two-photon autofluorescence
}
V1_USED_CHANNELS = (1,)  # v1 used channel 1 only; 0, 2, 3 never used

# --- cohort facts asserted by the brief, verified by scripts/00_inventory.py -
EXPECTED_SRS_ONLY = ("A-4", "E-2")
EXPECTED_HE_ONLY = ("A-3",)
EXPECTED_N_PAIRED = 77

# --- cores excluded from all downstream work --------------------------------
# Excluded at the whole-core level, with the evidence in DECISION_LOG.md.
# Distinct from V1_DEFECTIVE_HE_CORES below, which v1 identified as *poorly
# stained* but still usable; these are unusable.
EXCLUDED_CORES = {
    # Stored across the full uint16 range where the rest of the cohort is
    # 12-bit-valued, and SATURATED over most of the tissue; pronounced
    # stitching-grid and TPF banding artefacts. Also the only core carrying an
    # anomalous 183-DPI resolution tag - two independent signs that it passed
    # through different processing.
    #
    # NOTE: this core originally also failed segmentation outright, and that
    # was given as the reason. The corner-based background estimator
    # (2026-09-22) fixed the segmentation, so segmentation is no longer the
    # justification. The exclusion stands on the saturation itself: intensity
    # information destroyed by clipping cannot be recovered, and this project
    # turns on absolute level.
    "A-6": "SRS saturated/rescaled to full uint16; intensity unrecoverable",
    # H&E is damaged - confirmed by the project owner as unusable for study.
    # Independently visible in QC: pale, non-uniform background with a hard
    # step, and the core clipped by the frame.
    "A-1": "H&E damaged; disregard for study (project owner)",
}


def usable_cores(cores) -> list[str]:
    """Filter an iterable of core IDs down to those admissible downstream."""
    return [c for c in cores if c not in EXCLUDED_CORES]

# Cores with genuinely defective H&E staining (HANDOVER.md section 7).
# Detected by H-area fraction per StarDist-detected nucleus, NOT by H-channel
# percentiles, which conflate stain quality with tissue composition.
V1_DEFECTIVE_HE_CORES = ("C-7", "H-1", "B-7")
# Stroma-rich but normally stained - low nuclear density is real tissue, not an
# artefact. Must NOT be gated out.
V1_STROMA_RICH_CORES = ("G-1", "G-9")

# --- derived geometry (stages 1-2) ------------------------------------------
# Single source of truth. Duplicating these across scripts let 06 and 13 drift
# apart by a re-fit; they are small differences but there is no reason to carry
# two values for one physical constant.
#
# H&E native px per SRS native px, pooled over clean cores (stage 1).
COHORT_SCALE = 1.02317          # +/- 0.00098 (SEM, n=57)
# H&E is mirrored left-right relative to SRS; rotation is zero to within the
# 1 deg measurement limit (stage 2: 71/75 cores, p = 6.8e-17).
COHORT_MIRROR = True
# Implied pixel sizes, assuming a nominal 1 mm punch. Indicative only - punch
# diameter itself varies (CV 2.7%), so it is not a precision standard. The
# RATIO above is the well-determined quantity, and only the ratio is needed.
SRS_UM_PER_PX = 0.22701
HE_UM_PER_PX = SRS_UM_PER_PX / COHORT_SCALE

# --- StarDist protocol - must match v1 exactly for like-for-like comparison --
# HANDOVER.md section 6.
STARDIST_MODEL = "2D_versatile_he"
STARDIST_NORM_PERCENTILES = (1.0, 99.8)
STARDIST_MIN_NUCLEUS_AREA_PX = 10


# --- host settings used for the paper (24 physical cores / 32 threads CPU, 128 GB RAM, ~17 GB GPU) ---
# StarDist runs on CPU only and training is GPU-bound, so evaluation runs concurrently with
# training but is capped: 4 parallel evaluations slowed training from 296 s to 640 s per
# epoch by starving the dataloader workers; 2 parallel x 3 threads was measured safe.
HOST_PHYSICAL_CORES = 24
HOST_LOGICAL_CORES = 32
HOST_P_CORES = 8
HOST_E_CORES = 16
STARDIST_MAX_PARALLEL = 2
STARDIST_THREADS_PER_PROC = 3
# Leave the P-cores for the training dataloader; cap workers well short of 32.
DATALOADER_WORKERS = 8


def ensure_work_dirs() -> None:
    """Create the working tree on demand."""
    for d in (WORK_ROOT, PATCH_DIR, CHECKPOINT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ensure_project_dirs() -> None:
    for d in (RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
