"""
s18lib -- shared, detector-independent measurement code for the s18 metric
screen. Imports only numpy / scipy / scikit-image / pandas so it runs in
srs-analysis (StarDist), vhe-detectors and vhe-cellvit alike.

Every image (real original, real round trip, fake) is measured the SAME way:
from its RGB. Stain OD = skimage separate_stains with v1's calibrated
hed_from_rgb (the exact call inside v1 rgb_to_he, before the [-1, 1]
rescale). For fakes this reproduces the generator's own H to within 0.6%
(checked 2026-10-01, D-29). Note: rgb_to_he clips H at the calibration's
global range (0.125), so fakes and round-trip reals cannot exceed it while
original reals can; that is why both real baselines are reported.

Units: the 512 grid is 100 um, so 1 px = 0.1953 um, 1 px^2 = 0.03815 um^2.
"""

import os
import functools

import numpy as np
import pandas as pd

UM_PER_PX = 100.0 / 512
UM2_PER_PX = UM_PER_PX ** 2
MIN_AREA = 10                    # v1 protocol, px on the 512 grid
TISSUE_OD_SUM = 0.20             # summed RGB optical density above which a pixel is tissue
TISSUE_MIN_OBJ = 200             # px; tissue specks / holes smaller than this are cleaned
EDGE_PX = 20                     # territories of nuclei closer than this to the border are edge-biased
TERRITORY_STRIDE = 2


@functools.lru_cache(maxsize=1)
def hed_from_rgb():
    import countlib as cl
    return np.load(cl.CALIBRATION_PATH)["hed_from_rgb"]


def read_png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def write_png(path, rgb):
    from PIL import Image
    Image.fromarray(rgb).save(path, compress_level=1)


def stain_od(rgb):
    """(H, E) calibrated optical density, float32 (h, w)."""
    from skimage.color import separate_stains
    hed = separate_stains(rgb, hed_from_rgb())
    return hed[..., 0].astype(np.float32), hed[..., 1].astype(np.float32)


def tissue_mask(rgb):
    """Stained pixels: summed RGB OD > TISSUE_OD_SUM after light smoothing;
    identical rule for real and fake."""
    from scipy.ndimage import gaussian_filter
    od = -np.log10((rgb.astype(np.float32) + 1.0) / 256.0).sum(-1)
    m = gaussian_filter(od, 2) > TISSUE_OD_SUM
    m = _drop_small(m, TISSUE_MIN_OBJ)
    return ~_drop_small(~m, TISSUE_MIN_OBJ)


def _drop_small(mask, min_px):
    """Remove 8-connected components smaller than min_px (same rule in every
    scikit-image version)."""
    from scipy.ndimage import label
    lab, n = label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    sz = np.bincount(lab.ravel())
    keep = sz >= min_px; keep[0] = False
    return keep[lab]


def nucleus_table(lab, H, E, shape=True):
    """One row per instance with area >= MIN_AREA: centroid (px, 512 grid),
    area (um^2), solidity, eccentricity, mean H and E OD inside the nucleus."""
    from skimage.measure import regionprops_table
    lab = np.asarray(lab)
    if lab.max() == 0:
        return pd.DataFrame(columns=["nid", "cy", "cx", "area_um2", "solidity", "eccentricity", "mean_H", "mean_E"])
    props = ["label", "centroid", "area"] + (["solidity", "eccentricity"] if shape else [])
    t = pd.DataFrame(regionprops_table(lab, properties=props))
    t = t.rename(columns={"label": "nid", "centroid-0": "cy", "centroid-1": "cx"})
    flat = lab.ravel(); n = int(flat.max()) + 1
    cnt = np.bincount(flat, minlength=n)
    sH = np.bincount(flat, weights=H.ravel(), minlength=n); sE = np.bincount(flat, weights=E.ravel(), minlength=n)
    t["mean_H"] = sH[t.nid.values] / np.maximum(cnt[t.nid.values], 1)
    t["mean_E"] = sE[t.nid.values] / np.maximum(cnt[t.nid.values], 1)
    t = t[t.area >= MIN_AREA].copy()
    t["area_um2"] = t.pop("area") * UM2_PER_PX
    if not shape:
        t["solidity"] = np.nan; t["eccentricity"] = np.nan
    cols = ["nid", "cy", "cx", "area_um2", "solidity", "eccentricity", "mean_H", "mean_E"]
    return t[cols].reset_index(drop=True)


def spacing(cy, cx, tissue):
    """Mean nearest-neighbour distance (um) and median 'cell territory' (um^2):
    discrete Voronoi of tissue pixels around nucleus centroids (stride 2),
    median over nuclei at least EDGE_PX from the border (all nuclei if none)."""
    from scipy.spatial import cKDTree
    n = len(cy)
    if n < 2:
        return np.nan, np.nan
    pts = np.c_[cy, cx]
    tr = cKDTree(pts)
    d, _ = tr.query(pts, k=2)
    nnd = float(d[:, 1].mean() * UM_PER_PX)
    s = TERRITORY_STRIDE
    yy, xx = np.nonzero(tissue[::s, ::s])
    if len(yy) == 0:
        return nnd, np.nan
    _, owner = tr.query(np.c_[yy * s + s / 2, xx * s + s / 2])
    area = np.bincount(owner, minlength=n) * (s * s) * UM2_PER_PX
    inner = (cy >= EDGE_PX) & (cy < 512 - EDGE_PX) & (cx >= EDGE_PX) & (cx < 512 - EDGE_PX)
    terr = float(np.median(area[inner] if inner.any() else area))
    return nnd, terr


def stain_patch(rgb):
    """Detector-free patch stain balance (M5) and paleness, in tissue only."""
    H, E = stain_od(rgb)
    t = tissue_mask(rgb)
    rec = dict(tissue_frac=float(t.mean()))
    if t.sum() < 100:
        rec.update(mean_H_tissue=np.nan, mean_E_tissue=np.nan, H_frac=np.nan, H_p99_tissue=np.nan)
    else:
        mh, me = float(H[t].mean()), float(E[t].mean())
        rec.update(mean_H_tissue=mh, mean_E_tissue=me, H_frac=mh / max(mh + me, 1e-9),
                   H_p99_tissue=float(np.percentile(H[t], 99)))
    return rec, H, E, t


def patch_from_table(tab, tissue):
    """Patch-level M1-M4 summaries from a nucleus table."""
    n = len(tab)
    rec = dict(n=n)
    rec["med_area_um2"] = float(tab.area_um2.median()) if n else np.nan
    rec["med_solidity"] = float(tab.solidity.median()) if n else np.nan
    rec["med_eccentricity"] = float(tab.eccentricity.median()) if n else np.nan
    rec["med_nuc_H"] = float(tab.mean_H.median()) if n else np.nan
    rec["mean_nnd_um"], rec["med_territory_um2"] = spacing(tab.cy.values, tab.cx.values, tissue) if n else (np.nan, np.nan)
    return rec


def measure_image(rgb, lab, shape=True):
    """Full measurement of one image given its label image. Returns
    (patch_record, nucleus_table)."""
    srec, H, E, t = stain_patch(rgb)
    tab = nucleus_table(lab, H, E, shape=shape)
    rec = patch_from_table(tab, t)
    rec.update(srec)
    return rec, tab
