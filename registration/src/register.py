"""Map H&E into the SRS pixel grid.

All four degrees of freedom were determined in stages 1-2:

  * **scale**     one cohort constant, 1.02459 +/- 0.00107 H&E px per SRS px
  * **rotation**  none, to within the 1 deg measurement limit
  * **mirror**    one global left-right flip of the H&E
  * **translation** per core, from the fitted punch-circle centres

so the transform is a similarity with no rotation term, and the only per-core
quantity is a translation. H&E is warped into the SRS grid rather than the
reverse because SRS is the model input: keeping it unresampled avoids
interpolating the data the network actually reads.

Coordinate convention
---------------------
The mirror is applied to the H&E array as `fliplr`, which maps x -> W-1-x and
therefore moves the fitted centre to `W-1-cx_he`. With `scale` defined as H&E
px per SRS px, a displacement of `d` SRS pixels from the SRS centre corresponds
to `d * scale` H&E pixels from the mirrored H&E centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Transform:
    """H&E -> SRS similarity, no rotation."""
    core: str
    scale: float          # H&E native px per SRS native px
    mirror: bool          # fliplr the H&E
    srs_cy: float         # fitted punch centre, SRS native px
    srs_cx: float
    he_cy: float          # fitted punch centre, H&E native px (pre-mirror)
    he_cx: float
    he_width: int         # needed to mirror the centre
    dy: float = 0.0       # refinement, SRS px, added to the SRS-side centre
    dx: float = 0.0

    def he_centre_mirrored(self) -> tuple[float, float]:
        cx = (self.he_width - 1 - self.he_cx) if self.mirror else self.he_cx
        return self.he_cy, cx


def he_coords_for_srs_grid(t: Transform, out_shape: tuple[int, int]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """H&E sampling coordinates for every pixel of an SRS-grid output.

    Returns (ys, xs) into the *mirrored* H&E array.
    """
    h, w = out_shape
    ys_s, xs_s = np.mgrid[0:h, 0:w].astype(np.float32)
    # displacement from the SRS punch centre, including refinement
    dy = ys_s - (t.srs_cy + t.dy)
    dx = xs_s - (t.srs_cx + t.dx)
    hcy, hcx = t.he_centre_mirrored()
    return hcy + dy * t.scale, hcx + dx * t.scale


def warp_he_to_srs(rgb: np.ndarray, t: Transform, out_shape: tuple[int, int],
                   order: int = 1, cval: float = 255.0) -> np.ndarray:
    """Resample an H&E RGB image onto the SRS pixel grid.

    `cval=255` fills outside-the-slide regions with white, matching real slide
    background, so downstream optical-density calculations see background
    rather than a spurious dark border.
    """
    src = rgb[:, ::-1] if t.mirror else rgb
    ys, xs = he_coords_for_srs_grid(t, out_shape)
    chans = [ndi.map_coordinates(src[..., c].astype(np.float32), [ys, xs],
                                 order=order, mode="constant", cval=cval)
             for c in range(src.shape[2])]
    return np.clip(np.stack(chans, axis=-1), 0, 255)


def warp_gray_to_srs(img: np.ndarray, t: Transform, out_shape: tuple[int, int],
                     order: int = 1, cval: float = np.nan) -> np.ndarray:
    """Resample a single-channel H&E-space map onto the SRS pixel grid."""
    src = img[:, ::-1] if t.mirror else img
    ys, xs = he_coords_for_srs_grid(t, out_shape)
    return ndi.map_coordinates(src.astype(np.float32), [ys, xs],
                               order=order, mode="constant", cval=cval)


def soft_disc(shape: tuple[int, int], cy: float, cx: float, r: float,
              inner: float = 0.80, outer: float = 0.95) -> np.ndarray:
    """Radial cosine taper: 1 inside `inner*r`, 0 beyond `outer*r`.

    A hard disc mask puts a step edge into both maps, and that edge - being the
    strongest feature either map contains - dominates a cross-correlation and
    swamps the interior structure the shift is supposed to be measured from.
    Tapering removes the step while keeping the interior intact.
    """
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
    d = np.hypot(gy - cy, gx - cx) / max(r, 1e-6)
    w = np.clip((outer - d) / max(outer - inner, 1e-6), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * w)).astype(np.float32)


def refine_shift(a: np.ndarray, b: np.ndarray, max_shift: int
                 ) -> tuple[float, float, float]:
    """Shift of `b` relative to `a`, searched only within +/- max_shift px.

    Returns (dy, dx, peak_correlation).

    Bounding the search is essential here. The two maps have already been
    aligned on their fitted punch centres, so the true peak lies near zero and
    the correlation surface is broad and shallow around it. An unbounded search
    over a shallow surface will happily return a noise peak on the far side of
    the core - observed as "residuals" of up to 650 um on a 1000 um punch.
    The bound encodes what stage 1 already established: circle centres are
    accurate to a few percent of the radius, not to half a core.
    """
    a = np.nan_to_num(a, nan=0.0).astype(np.float32)
    b = np.nan_to_num(b, nan=0.0).astype(np.float32)
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return 0.0, 0.0, float("nan")

    cc = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    cc = np.fft.fftshift(cc) / denom
    c0y, c0x = a.shape[0] // 2, a.shape[1] // 2

    m = max_shift
    win = cc[c0y - m:c0y + m + 1, c0x - m:c0x + m + 1]
    if win.size == 0:
        return 0.0, 0.0, float("nan")
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    peak = float(win[iy, ix])

    # parabolic subpixel refinement on each axis
    def sub(v0, vm, vp):
        d = vm - 2.0 * v0 + vp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (vm - vp) / d

    sy = sx = 0.0
    if 0 < iy < win.shape[0] - 1:
        sy = sub(win[iy, ix], win[iy - 1, ix], win[iy + 1, ix])
    if 0 < ix < win.shape[1] - 1:
        sx = sub(win[iy, ix], win[iy, ix - 1], win[iy, ix + 1])

    return float(iy - m + sy), float(ix - m + sx), peak
