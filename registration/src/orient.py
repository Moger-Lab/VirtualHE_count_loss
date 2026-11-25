"""Empirical determination of relative orientation between SRS and H&E.

HANDOVER.md section 2 requires flips to be determined empirically for both axes
and both modalities, and specifically flags that v1 never tested a vertical flip
of the H&E.

Why this is a separate stage from outline alignment
---------------------------------------------------
DECISION_LOG.md stage 1: a circular outline is rotationally symmetric, so it
fixes scale and centroid and *nothing else*. Outline alignment cannot recover
rotation or resolve a mirror. Orientation has to come from internal structure.

The method
----------
The stage-1 punch-circle fit already gives each core's centre and radius in its
own modality. Resampling both images onto a polar grid defined by their own
fitted circle therefore removes translation and scale exactly, leaving only two
unknowns: a rotation and a possible mirror.

In polar coordinates those become trivial:

  * rotation by phi  ->  a circular shift along the angular axis
  * mirror           ->  reversal of the angular axis

So all rotations are evaluated at once by circular cross-correlation along
theta (via FFT), and the mirror is tested by repeating on the angle-reversed
map. Two FFT correlations per core give the best rotation under each parity and
the score difference between them.

What the correlation is computed on
-----------------------------------
Gross architecture - where tissue is and where lumina, fat and empty space are -
is conserved between parallel sections even though individual cells are not
(HANDOVER.md section 2: R^2 rises from 0.16 at 6 um to 0.53 at 100 um). So the
signal used is tissue density, smoothed to a scale where correspondence
physically exists, not nuclear detail.

Each polar ring is normalised independently. Without that, the radial intensity
profile - which both modalities share simply because a core is denser in the
middle and fades at the rim - dominates the correlation and swamps the angular
information that actually determines rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

N_RADIUS = 72
N_THETA = 360
R_INNER = 0.08     # skip the centre: tiny area, and angularly ill-defined
R_OUTER = 0.94     # skip the rim: segmentation is least certain there


@dataclass
class OrientResult:
    core: str
    best_angle_deg: float      # rotation applied to H&E to match SRS
    best_score: float
    best_mirrored: bool
    score_direct: float        # best score without mirroring
    score_mirrored: float      # best score with mirroring
    margin: float              # mirrored - direct; sign gives the verdict
    angle_direct_deg: float
    angle_mirrored_deg: float


def to_polar(img: np.ndarray, cy: float, cx: float, r: float,
             n_r: int = N_RADIUS, n_t: int = N_THETA) -> np.ndarray:
    """Resample `img` onto a (radius, theta) grid set by a fitted circle.

    Sampling each modality by its *own* fitted circle removes translation and
    scale, so the returned maps are directly comparable.
    """
    rr = np.linspace(R_INNER * r, R_OUTER * r, n_r)
    tt = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
    R, T = np.meshgrid(rr, tt, indexing="ij")
    ys = cy + R * np.sin(T)
    xs = cx + R * np.cos(T)
    return ndi.map_coordinates(img.astype(np.float32), [ys, xs],
                               order=1, mode="constant", cval=np.nan)


def normalise_rings(polar: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance each radius ring independently.

    Removes the shared radial profile, which carries no angular information but
    would otherwise dominate the correlation.
    """
    out = np.zeros_like(polar, dtype=np.float32)
    for i in range(polar.shape[0]):
        ring = polar[i]
        good = np.isfinite(ring)
        if good.sum() < ring.size // 2:
            continue
        m = ring[good].mean()
        s = ring[good].std()
        if s <= 1e-8:
            continue
        r = np.where(good, (ring - m) / s, 0.0)
        out[i] = r
    return out


def _circular_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular cross-correlation along the angular axis, summed over rings.

    Rings are weighted by radius so that each unit of *area* contributes
    equally; without it the innermost rings, which have the fewest independent
    samples, would count as much as the outermost.
    """
    n_r, n_t = a.shape
    w = np.linspace(R_INNER, R_OUTER, n_r).astype(np.float32)
    fa = np.fft.rfft(a, axis=1)
    fb = np.fft.rfft(b, axis=1)
    cc = np.fft.irfft(fa * np.conj(fb), n=n_t, axis=1)
    return (cc * w[:, None]).sum(axis=0) / (w.sum() * n_t)


def determine(srs_map: np.ndarray, he_map: np.ndarray, core: str
              ) -> OrientResult:
    """Best rotation and mirror parity aligning the H&E map to the SRS map."""
    a = normalise_rings(srs_map)
    b = normalise_rings(he_map)

    cc_direct = _circular_xcorr(a, b)
    # A mirror in Cartesian space is a reversal of the angular axis.
    cc_mirror = _circular_xcorr(a, b[:, ::-1])

    i_d = int(np.argmax(cc_direct))
    i_m = int(np.argmax(cc_mirror))
    s_d = float(cc_direct[i_d])
    s_m = float(cc_mirror[i_m])

    step = 360.0 / srs_map.shape[1]
    ang_d = i_d * step
    ang_m = i_m * step

    mirrored = s_m > s_d
    return OrientResult(
        core=core,
        best_angle_deg=ang_m if mirrored else ang_d,
        best_score=max(s_m, s_d),
        best_mirrored=mirrored,
        score_direct=s_d,
        score_mirrored=s_m,
        margin=s_m - s_d,
        angle_direct_deg=ang_d,
        angle_mirrored_deg=ang_m,
    )
