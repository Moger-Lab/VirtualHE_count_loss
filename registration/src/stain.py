"""Haematoxylin/eosin separation, with the stain matrix fitted to this cohort.

HANDOVER.md section 3 step 6: "Stain calibration from scratch - H&E ->
haematoxylin/eosin separation. Fit the stain matrix from this cohort and
establish global (not per-patch) normalisation ranges."

Why fit rather than inherit
---------------------------
The standard Ruifrok/Johnston H&E vectors were derived from particular stains,
scanners and protocols. This cohort is commercially sourced TMA material of
visibly varying stain strength - three cores are known to be badly stained
(`C-7`, `H-1`, `B-7`). Using published vectors would fold that mismatch into
every downstream density measurement. The vectors are recoverable from the data
itself, so they are.

Why normalisation must be global
--------------------------------
HANDOVER.md section 7: "Per-patch percentile normalisation destroys absolute
level." A patch of dense tumour and a patch of sparse stroma, each normalised to
its own percentiles, come out looking equally stained - which erases exactly the
density information the model has to learn to condition on. Concentration
ranges are therefore fitted once over the cohort and applied unchanged to every
patch.

Method
------
Macenko: optical density, discard near-background pixels, take the plane
spanned by the two leading singular vectors, and read the two stain vectors off
the angular extremes within that plane. Robust percentiles rather than true
extremes, so a handful of saturated or dust pixels cannot define a stain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ruifrok/Johnston reference vectors, used only as a sanity check on the fit -
# never as the operating values.
RUIFROK_H = np.array([0.650, 0.704, 0.286])
RUIFROK_E = np.array([0.072, 0.990, 0.105])


@dataclass
class StainMatrix:
    """Two unit OD vectors, haematoxylin first."""
    h: np.ndarray
    e: np.ndarray
    n_pixels: int
    beta: float
    alpha: float

    @property
    def matrix(self) -> np.ndarray:
        """(2, 3): rows are the stain vectors."""
        return np.stack([self.h, self.e])

    def angle_to(self, other_h, other_e) -> tuple[float, float]:
        """Angular deviation in degrees from a reference pair."""
        def ang(a, b):
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
        return ang(self.h, other_h), ang(self.e, other_e)


def rgb_to_od(rgb: np.ndarray, io: float = 255.0) -> np.ndarray:
    """Optical density. Input may be (..., 3) uint8 or float in [0, 255]."""
    x = np.clip(rgb.astype(np.float32), 1.0, io)
    return -np.log10(x / io)


def od_to_rgb(od: np.ndarray, io: float = 255.0) -> np.ndarray:
    return np.clip(io * np.power(10.0, -od), 0, 255)


def tissue_od_pixels(rgb: np.ndarray, beta: float = 0.15,
                     mask: np.ndarray | None = None) -> np.ndarray:
    """OD rows for pixels carrying stain, i.e. above the background floor."""
    od = rgb_to_od(rgb).reshape(-1, 3)
    keep = od.sum(axis=1) > beta * 3.0
    if mask is not None:
        keep &= mask.reshape(-1)
    return od[keep]


def fit_macenko(od: np.ndarray, beta: float = 0.15, alpha: float = 1.0
                ) -> StainMatrix:
    """Fit two stain vectors from pooled OD pixels.

    `alpha` is the robust percentile used for the angular extremes; 1.0 means
    the 1st and 99th. True extremes would let a single saturated or dust pixel
    define a stain vector.
    """
    if od.shape[0] < 1000:
        raise ValueError(f"too few tissue pixels to fit: {od.shape[0]}")

    # Plane of greatest variation in OD space.
    _, _, vt = np.linalg.svd(od - od.mean(axis=0, keepdims=True),
                             full_matrices=False)
    plane = vt[:2]                       # (2, 3)

    proj = od @ plane.T                  # (N, 2)
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    # Centre the angular frame on the data's circular mean before taking
    # percentiles.
    #
    # The sign of an SVD singular vector is arbitrary, so the stain cone can
    # land anywhere on the circle - including straddling arctan2's branch cut
    # at +/-pi. When it does, ordinary percentiles are nonsense: the 1st and
    # 99th percentiles come out at -179.9 and +179.9 degrees, which are
    # *adjacent* angles, not opposite extremes. Observed directly - a 30-core
    # pool reported a stain separation of 0.0 degrees and an angular span of
    # exactly 360, the signature of a wrapped distribution.
    #
    # Rotating to the circular mean puts the cone in the middle of the range,
    # where percentiles mean what they are supposed to mean, whatever sign the
    # SVD happened to choose.
    mu = float(np.arctan2(np.sin(phi).mean(), np.cos(phi).mean()))
    phi_c = np.angle(np.exp(1j * (phi - mu)))        # wrapped to [-pi, pi]
    lo = float(np.percentile(phi_c, alpha)) + mu
    hi = float(np.percentile(phi_c, 100.0 - alpha)) + mu

    v1 = plane.T @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane.T @ np.array([np.cos(hi), np.sin(hi)])

    # A stain vector is an absorbance and must be non-negative; the SVD basis
    # carries an arbitrary sign.
    if v1.sum() < 0:
        v1 = -v1
    if v2.sum() < 0:
        v2 = -v2
    v1 = np.clip(v1, 0, None)
    v2 = np.clip(v2, 0, None)
    v1 /= max(np.linalg.norm(v1), 1e-9)
    v2 /= max(np.linalg.norm(v2), 1e-9)

    # Order by red absorbance. Haematoxylin renders blue-purple, so it absorbs
    # strongly in red (Ruifrok H has R = 0.650); eosin renders pink and barely
    # absorbs red at all (E has R = 0.072). The gap is large and unambiguous.
    h, e = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    return StainMatrix(h=h, e=e, n_pixels=int(od.shape[0]),
                       beta=beta, alpha=alpha)


def deconvolve(rgb: np.ndarray, sm: StainMatrix) -> np.ndarray:
    """Per-pixel haematoxylin and eosin concentrations. Returns (..., 2)."""
    shape = rgb.shape[:-1]
    od = rgb_to_od(rgb).reshape(-1, 3)
    # Least-squares against a 2-stain basis; H&E carries no third chromogen, so
    # a pseudo-inverse of the 2x3 matrix is the right operation - not the 3x3
    # inversion used when a residual "background" stain is modelled.
    conc = od @ np.linalg.pinv(sm.matrix)
    return np.clip(conc, 0, None).reshape(*shape, 2)
