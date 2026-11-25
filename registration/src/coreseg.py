"""Core footprint segmentation for both modalities.

HANDOVER.md section 2: neither modality is calibrated and no pixel-size metadata
survives in the files (confirmed in DECISION_LOG.md stage 0). The cores are the
same physical object in both modalities, so the outline is the only route to
scale.

What this module segments is the core **footprint** - the outer extent of the
tissue disc - not the tissue-signal region. Those differ: a core with tears,
lumina or low-signal stroma has interior gaps that belong to the footprint.
Interior holes are therefore filled without a size cap.

Revision history
----------------
v1 (first pass) used Otsu + size-capped hole fill + largest-component. QC showed
four failure modes, all fixed here:

1. **Interior holes survived** the 1%-of-frame cap (`A-1`'s tears). Now filled
   unconditionally via `binary_fill_holes`.
2. **Largest-component discarded fragments.** `A-2` is a genuinely fragmented
   core; keeping one piece silently dropped the rest, and the largest SRS
   fragment need not be the largest H&E fragment, so the resulting scale could
   compare different pieces of tissue. Now keeps every component above a
   fraction of the largest and reports the count.
3. **Clipping was flagged but not quantified.** `G-1`'s core runs off the SRS
   frame, making its diameter a lower bound and its scale an over-estimate. Now
   measured as the fraction of the footprint boundary lying on the frame edge.
4. **Otsu collapsed on faint H&E.** `A-1` is pale; a global two-class threshold
   put most of the core in the background class. Now background-referenced (see
   `segment_he`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.transform import resize


@dataclass
class CoreShape:
    """Footprint statistics of one segmented core."""
    core: str
    modality: str
    ds: int
    # footprint = union of kept fragments, interiors filled
    area_px_ds: float
    area_px_full: float
    equiv_diam_full: float       # equivalent-circle diameter of the footprint
    major_axis_full: float
    minor_axis_full: float
    perimeter_full: float
    eccentricity: float
    solidity: float              # footprint area / convex-hull area
    extent: float
    orientation_deg: float
    centroid_y_full: float
    centroid_x_full: float
    frac_of_frame: float
    # fragmentation
    n_fragments: int             # components kept
    largest_frag_share: float    # largest fragment / total footprint area
    # clipping - a core running off the frame has an under-measured diameter
    border_frac: float           # share of footprint boundary on the frame edge
    touches_border: bool
    # fitted punch circle - the physically meaningful core size
    circle_cy_full: float
    circle_cx_full: float
    circle_diam_full: float
    circle_rms_full: float       # fit residual; high => not a circular punch
    circle_rms_rel: float        # residual / radius, scale-free quality metric
    # threshold provenance
    threshold: float
    bg_level: float


def _fill_and_keep(mask: np.ndarray, rel_min: float = 0.05
                   ) -> tuple[np.ndarray, int, float]:
    """Fill interiors, then keep every fragment >= rel_min of the largest.

    A TMA core may be intact (one component) or broken into several. Both are
    real; discarding the smaller pieces loses tissue that exists in both
    sections and distorts any area-derived measurement.
    """
    lab, n = ndi.label(mask)
    if n == 0:
        return mask, 0, 0.0
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    biggest = areas.max()
    keep_ids = np.nonzero(areas >= rel_min * biggest)[0] + 1
    kept = np.isin(lab, keep_ids)
    kept = ndi.binary_fill_holes(kept)
    total = float(kept.sum())
    return kept, int(len(keep_ids)), float(biggest / total) if total else 0.0


def _border_fraction(mask: np.ndarray) -> float:
    """Share of the footprint's boundary that lies on the image edge.

    `border_value=0` matters: with `border_value=1` the erosion treats
    out-of-frame as foreground, so mask pixels on the image edge are classed as
    interior and excluded from the boundary - which let this ratio exceed 1.0
    (observed: 2.069 on `A-1`). With 0 they are correctly boundary pixels and
    the ratio is bounded by 1.
    """
    if not mask.any():
        return 0.0
    eroded = ndi.binary_erosion(mask, np.ones((3, 3)), border_value=0)
    boundary = mask & ~eroded
    n_boundary = int(boundary.sum())
    if n_boundary == 0:
        return 0.0
    edge = np.zeros_like(mask)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return float((boundary & edge).sum()) / n_boundary


def _fit_circle(ys: np.ndarray, xs: np.ndarray
                ) -> tuple[float, float, float, float]:
    """Geometric least-squares circle fit. Returns (cy, cx, r, rms_residual).

    TMA cores are circular punches, so the punch circle - not the tissue
    outline - is the physically meaningful size. Fitting it rather than
    measuring the tissue footprint solves three problems at once:

      * **interior gaps and bays** (a tear that opens to the core edge is not a
        topological hole, so hole-filling cannot recover it - `D-3`'s SRS left
        margin) no longer shrink the estimate;
      * **clipping** stops biasing the result, because a circle is recoverable
        from a partial arc - which is what `G-1` presents;
      * **fragmentation** is tolerated as long as the pieces have not moved,
        and the residual reports it when they have.

    Kasa algebraic fit for initialisation, then geometric refinement. Kasa
    alone is biased on short arcs, which is precisely the clipped case.
    """
    from scipy.optimize import least_squares

    y = ys.astype(np.float64)
    x = xs.astype(np.float64)
    if y.size < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Kasa: x^2 + y^2 + Dx + Ey + F = 0
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan"), float("nan")
    cx0, cy0 = -sol[0] / 2.0, -sol[1] / 2.0
    r0sq = cx0 ** 2 + cy0 ** 2 - sol[2]
    r0 = float(np.sqrt(r0sq)) if r0sq > 0 else float(np.hypot(x.std(), y.std()))

    def resid(p):
        cx, cy, r = p
        return np.hypot(x - cx, y - cy) - r

    try:
        out = least_squares(resid, [cx0, cy0, r0], method="lm", max_nfev=2000)
        cx, cy, r = out.x
        rms = float(np.sqrt(np.mean(out.fun ** 2)))
    except Exception:  # noqa: BLE001 - fall back to the algebraic fit
        cx, cy, r, rms = cx0, cy0, r0, float("nan")

    return float(cy), float(cx), float(abs(r)), rms


def _outer_boundary_points(foot: np.ndarray, edge_margin: int = 2,
                           exclude: np.ndarray | None = None,
                           exclude_dilate: int = 3
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Convex-hull boundary points, dropping any that are not real punch edge.

    Two kinds of boundary point are artefacts rather than punch rim, and both
    bias a circle fit if kept:

    * **frame edge** - an artefact of the field of view. Dragging the fit
      toward a frame corner.
    * **`exclude` region** - typically un-scanned blank canvas. Where a scan is
      cropped, the tissue stops along a straight line *inside* the frame. That
      line is a chord of the punch, not an arc of it, and a chord pulls the
      fitted centre toward the crop and shrinks the radius. Observed on seven
      cores, six of them A-row, with residuals of 26-67 um afterwards.

    Dropping both leaves only genuine rim, from which a circle is recoverable
    even when a substantial arc is missing.
    """
    hull = morphology.convex_hull_image(foot)
    eroded = ndi.binary_erosion(hull, np.ones((3, 3)), border_value=0)
    ys, xs = np.nonzero(hull & ~eroded)
    h, w = foot.shape
    keep = ((ys >= edge_margin) & (ys < h - edge_margin)
            & (xs >= edge_margin) & (xs < w - edge_margin))

    if exclude is not None and exclude.any():
        bad = ndi.binary_dilation(exclude, np.ones((3, 3)),
                                  iterations=max(1, exclude_dilate))
        keep &= ~bad[ys, xs]

    return ys[keep], xs[keep]


def blank_canvas(rgb_small: np.ndarray, level: float = 250.0,
                 texture: float = 0.6) -> np.ndarray:
    """Un-scanned canvas: pure white with no local texture.

    Distinguished from real slide background, which is off-white and carries
    fine scanner noise. Pure, texture-free white is region the scanner never
    covered.
    """
    bright = rgb_small.min(axis=-1) >= level
    g = rgb_small.mean(axis=-1)
    var = ndi.uniform_filter(g ** 2, 5) - ndi.uniform_filter(g, 5) ** 2
    return bright & (np.sqrt(np.clip(var, 0, None)) < texture)


def _corner_region(shape: tuple[int, int]) -> np.ndarray:
    """The frame corners - outside the largest inscribed circle.

    A TMA core is a disc. Whatever its size or position, the corners of the
    square frame lie outside the largest circle that fits in it, so they are
    background by construction. This is 1 - pi/4 = 21.5% of the frame, which is
    ample for a robust estimate.
    """
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    r = min(h, w) / 2.0
    return ((gy - h / 2.0) ** 2 + (gx - w / 2.0) ** 2) > r ** 2


def _robust_background(x: np.ndarray) -> tuple[float, float]:
    """Background level and spread, estimated from the frame corners.

    CORRECTED 2026-09-22. This previously took the level from the 2nd
    percentile of the whole image and the spread from everything below the
    25th - which silently assumes background occupies at least a quarter of the
    frame. Where tissue fills the frame that assumption fails, and the "tail"
    is made of tissue, so the spread is enormously overestimated and the
    threshold lands in the middle of the tissue:

        core   bg       spread    threshold
        D-3    0.0022   0.0051    0.033   (background-dominated frame, fine)
        B-6    0.0140   0.0325    0.209
        B-9    0.0237   0.0974    0.608   <- mask collapsed to 3.5% of frame

    Sampling the corners instead makes the estimate independent of how much
    tissue is present, which is the property the whole approach was supposed to
    have and the reason it was preferred over Otsu in the first place.
    """
    corner = x[_corner_region(x.shape)]
    if corner.size < 64:
        corner = x.ravel()
    lo = float(np.percentile(corner, 20.0))
    spread = float(np.median(np.abs(corner - lo)))
    return lo, max(spread, 1e-6)


# Absolute threshold floors, in the units of their respective signals.
# Slide glass sits at OD-sum ~0.02-0.05; stained tissue, even faintly, exceeds
# 0.15. The floor is applied *relative to the measured background*, so a scan
# with an offset or uneven background is still handled.
OD_FLOOR = 0.12
CHROMA_FLOOR = 0.06
SRS_FLOOR = 0.020
# Coverage above which the threshold is assumed to have leaked into background.
#
# CORRECTED 2026-09-22. This was 0.82, on the reasoning that a disc inscribed in
# a square reaches only pi/4 = 0.785 so anything above ~0.82 must be leakage.
# That reasoning was wrong in practice: cores routinely run off their frame
# (71 of 75 SRS punches are clipped), and a clipped core's tissue legitimately
# covers more than 0.82. The guard then fired on perfectly good images and
# escalated the threshold until it destroyed them - `B-6` ended with an H&E
# threshold of 0.667 against a cohort-normal 0.187, leaving a mask of scattered
# blobs and a valid fraction of 0.085.
#
# The guard's real job is catching catastrophic leakage, where coverage goes to
# ~1.0. 0.93 does that while leaving legitimately frame-filling cores alone.
MAX_COVERAGE = 0.93
# Gentler escalation, for the same reason: 1.6x overshot in a single step and
# never came back.
COVERAGE_ESCALATION = 1.25


def _threshold_with_coverage_guard(od_total, chroma, od_thr, ch_thr,
                                   od_bg, ch_bg, max_cov: float = MAX_COVERAGE,
                                   max_iter: int = 12) -> np.ndarray:
    """Threshold, escalating if the result covers implausibly much of the frame.

    A guard rather than a tuning knob: it only ever fires when the mask is
    already known to be wrong on geometric grounds.
    """
    od_t, ch_t = od_thr, ch_thr
    mask = (od_total > od_t) | (chroma > ch_t)
    for _ in range(max_iter):
        if mask.mean() <= max_cov:
            break
        od_t = od_bg + (od_t - od_bg) * COVERAGE_ESCALATION
        ch_t = ch_bg + (ch_t - ch_bg) * COVERAGE_ESCALATION
        mask = (od_total > od_t) | (chroma > ch_t)
    return mask


def _downsample(img: np.ndarray, ds: int) -> np.ndarray:
    y, x = img.shape[:2]
    shape = (y // ds, x // ds) + img.shape[2:]
    return resize(img.astype(np.float32), shape, order=1,
                  anti_aliasing=True, preserve_range=True)


# Channels usable as a tissue-presence indicator.
#
# Per the project owner: SHG (ch0) and TPF (ch3) arise only from *particular*
# structures - fibrillar collagen and autofluorescent material - so they report
# tissue *type*, not tissue *presence*, and their outlines are correspondingly
# incomplete (measured zero-fractions 0.79 and 0.49). SRS (ch1) and especially
# CARS (ch2) are generated throughout the core: CARS carries non-resonant
# background from any material present, which is exactly the "is there tissue
# here" signal an outline needs.
#
# Averaging all four - the first implementation - therefore pulled the outline
# inward wherever tissue lacked collagen or autofluorescence.
#
# All four channels derive from the same focal spot, detected in parallel, so
# they are registered to each other at pixel level. Channel choice here is
# purely about which signal reports tissue presence; no inter-channel
# alignment is implied or needed.
TISSUE_CHANNELS = (2,)      # CARS
TISSUE_CHANNELS_FALLBACK = (2, 1)   # CARS + fsSRS


def segment_srs(vol: np.ndarray, ds: int = 8, k: float = 6.0,
                channels: tuple[int, ...] = TISSUE_CHANNELS
                ) -> tuple[np.ndarray, float, float]:
    """Footprint mask from a 4-channel SRS volume. Returns (mask, thr, bg).

    `channels` selects the tissue-presence signal; see TISSUE_CHANNELS. Several
    channels are combined by per-channel robust normalisation then pixelwise
    max, so tissue is admitted where *any* selected channel sees it.

    The threshold is referenced to the background rather than fitted by Otsu.
    CARS intensity varies strongly between tissue types within a single core,
    so a two-class split on the whole histogram can place a weakly-responding
    tissue type in the background class; a background-referenced threshold is
    indifferent to how the foreground is distributed.
    """
    if vol.ndim != 3:
        raise ValueError(f"expected (C, Y, X), got {vol.shape}")
    if vol.shape[0] != 4 and vol.shape[-1] == 4:
        vol = np.moveaxis(vol, -1, 0)

    sel = [c for c in channels if c < vol.shape[0]]
    if not sel:
        raise ValueError(f"no usable channels in {channels} for {vol.shape}")

    planes = []
    for c in sel:
        s = _downsample(vol[c], ds)
        lo, hi = np.percentile(s, (1.0, 99.5))
        planes.append(np.clip((s - lo) / (hi - lo), 0, 1) if hi > lo
                      else np.zeros_like(s))
    combined = filters.gaussian(np.maximum.reduce(planes), sigma=2.0)

    bg, spread = _robust_background(combined)
    thr = max(bg + k * spread, bg + SRS_FLOOR)   # floor: see OD_FLOOR
    mask = combined > thr
    for _ in range(12):                          # same coverage guard as H&E
        if mask.mean() <= MAX_COVERAGE:
            break
        thr = bg + (thr - bg) * COVERAGE_ESCALATION
        mask = combined > thr
    mask = morphology.closing(mask, morphology.disk(
        max(3, int(0.012 * min(combined.shape)))))
    mask = morphology.remove_small_objects(mask, max_size=mask.size // 2000)
    return mask, float(thr), float(bg)


def segment_he(rgb: np.ndarray, ds: int = 8, k: float = 6.0
               ) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Footprint mask from an H&E RGB image. Returns (mask, thr, bg).

    Uses optical density OR chroma. Slide background is bright *and*
    achromatic; tissue is darker *and* coloured. A faintly stained core - `A-1`
    is nearly colourless in places - can fail a darkness test while still
    carrying clear pink/purple chroma, so either signal admits tissue.
    """
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"expected (Y, X, 3), got {rgb.shape}")
    small = _downsample(rgb[..., :3], ds)

    od = -np.log10(np.clip(small, 1.0, 255.0) / 255.0)
    od_total = filters.gaussian(od.sum(axis=-1), sigma=2.0)

    # chroma: distance from the achromatic axis
    chroma = filters.gaussian(
        small.max(axis=-1) - small.min(axis=-1), sigma=2.0) / 255.0

    od_bg, od_spread = _robust_background(od_total)
    ch_bg, ch_spread = _robust_background(chroma)

    # Absolute floors are essential, not belt-and-braces. On a cleanly scanned
    # slide the background is so uniform that its spread is near zero, so
    # `bg + k*spread` collapses onto the background level itself and admits the
    # entire frame. Observed on `G-1`, whose footprint then covered the whole
    # image and left the circle fit only near-collinear boundary points, which
    # it "fitted" as a circle of radius 1.5e6 px.
    od_thr = max(od_bg + k * od_spread, od_bg + OD_FLOOR)
    ch_thr = max(ch_bg + k * ch_spread, ch_bg + CHROMA_FLOOR)

    mask = _threshold_with_coverage_guard(
        od_total, chroma, od_thr, ch_thr, od_bg, ch_bg)

    mask = morphology.closing(mask, morphology.disk(
        max(3, int(0.012 * min(od_total.shape)))))
    mask = morphology.remove_small_objects(mask, max_size=mask.size // 2000)
    return mask, float(od_thr), float(od_bg), blank_canvas(small)


def describe(mask: np.ndarray, core: str, modality: str, ds: int,
             thr: float = float("nan"), bg: float = float("nan"),
             exclude: np.ndarray | None = None) -> CoreShape:
    """Footprint statistics. `mask` is the raw threshold output.

    `exclude` marks regions that must not contribute punch-rim points - in
    practice, un-scanned canvas from a cropped slide scan.
    """
    foot, n_frag, largest_share = _fill_and_keep(mask)
    if not foot.any():
        raise ValueError(f"{core}/{modality}: empty footprint")

    props = measure.regionprops(foot.astype(np.uint8))
    p = props[0]
    bfrac = _border_fraction(foot)

    by, bx = _outer_boundary_points(foot, exclude=exclude)
    cy, cx, r, rms = _fit_circle(by, bx)

    # Reject physically impossible fits rather than reporting them as numbers.
    # A near-collinear point set is fitted perfectly by an enormous circle
    # (observed on `G-1`: r = 7.6e5 px at rms 0.0000), which would otherwise
    # enter the scale pool as a confident-looking outlier.
    diag = float(np.hypot(*foot.shape))
    if not (np.isfinite(r) and by.size >= 50 and 0.05 * diag < r < 2.0 * diag):
        cy = cx = r = rms = float("nan")

    rms_rel = float(rms / r) if r and np.isfinite(r) and r > 0 else float("nan")

    return CoreShape(
        core=core, modality=modality, ds=ds,
        area_px_ds=float(p.area),
        area_px_full=float(p.area) * ds * ds,
        equiv_diam_full=float(p.equivalent_diameter_area) * ds,
        major_axis_full=float(p.axis_major_length) * ds,
        minor_axis_full=float(p.axis_minor_length) * ds,
        perimeter_full=float(p.perimeter) * ds,
        eccentricity=float(p.eccentricity),
        solidity=float(p.solidity),
        extent=float(p.extent),
        orientation_deg=float(np.degrees(p.orientation)),
        centroid_y_full=float(p.centroid[0]) * ds,
        centroid_x_full=float(p.centroid[1]) * ds,
        frac_of_frame=float(p.area) / foot.size,
        n_fragments=n_frag,
        largest_frag_share=round(largest_share, 4),
        border_frac=round(bfrac, 4),
        touches_border=bool(bfrac > 0.0),
        circle_cy_full=cy * ds, circle_cx_full=cx * ds,
        circle_diam_full=2.0 * r * ds,
        circle_rms_full=rms * ds,
        circle_rms_rel=round(rms_rel, 5),
        threshold=thr, bg_level=bg,
    )


def footprint(mask: np.ndarray) -> np.ndarray:
    """The filled, fragment-preserving footprint of a raw threshold mask."""
    return _fill_and_keep(mask)[0]


def shape_to_row(s: CoreShape) -> dict:
    return asdict(s)

