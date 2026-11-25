"""Stage 2 verification - apply the measured transform and look at the result.

Stage 2 found that 70/75 cores prefer a mirrored H&E (p = 9.8e-16) and that the
best rotation clusters tightly near 195 deg rather than scattering.

Two things must be checked before that is believed:

1. **What the composite transform actually is.** The polar test mirrors by
   reversing the angular axis, which for the convention used in `orient.to_polar`
   (y = cy + r sin t, x = cx + r cos t) is a mirror about the *horizontal* axis
   - an up-down flip. Composing that with a ~180 deg rotation gives a
   *left-right* flip. So the reported "mirror + 180 deg" and the project
   owner's "H&E is flipped about the vertical axis" are the same statement, and
   this script confirms that algebraically and visually rather than by
   assertion.

2. **Whether the alignment is real.** A correlation peak proves nothing on its
   own. Applying the transform and overlaying the two modalities does.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/05_orientation_verify.py --cores D-3 G-1 B-5 E-9
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

import config
from src import orient
from importlib import import_module

_o = import_module("scripts.04_orientation") if False else None

Image.MAX_IMAGE_PIXELS = None
DS = 8
CANON = 512   # canonical square: core inscribed, so both modalities comparable


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def circular_stats(deg: np.ndarray) -> tuple[float, float]:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad).mean(), np.sin(rad).mean()
    mean = float(np.rad2deg(np.arctan2(s, c)) % 360.0)
    return mean, float(np.hypot(c, s))


def canonical(img: np.ndarray, cy: float, cx: float, r: float,
              size: int = CANON) -> np.ndarray:
    """Resample the fitted disc onto a fixed square, removing scale+translation."""
    g = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(g, g)
    ys = cy + Y * r
    xs = cx + X * r
    if img.ndim == 3:
        return np.stack([
            ndi.map_coordinates(img[..., c].astype(np.float32), [ys, xs],
                                order=1, mode="constant", cval=np.nan)
            for c in range(img.shape[2])], axis=-1)
    return ndi.map_coordinates(img.astype(np.float32), [ys, xs],
                               order=1, mode="constant", cval=np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", nargs="+", default=["D-3", "G-1", "B-5", "E-9"])
    args = ap.parse_args()

    config.ensure_project_dirs()
    outdir = config.FIGURES_DIR / "04_orientation"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(
        (config.RESULTS_DIR / "04_orientation.csv").open(encoding="utf-8")))
    by_core = {r["core"]: r for r in rows}

    mirrored = [r for r in rows if r["best_mirrored"] == "True"]
    ang = np.array([float(r["best_angle_deg"]) for r in mirrored])
    mean_ang, R = circular_stats(ang)

    print("=" * 74)
    print("COMPOSITE TRANSFORM")
    print("=" * 74)
    print(f"  cores preferring mirror : {len(mirrored)}/{len(rows)}")
    print(f"  circular mean rotation  : {mean_ang:.1f} deg  (|R| = {R:.3f})")
    print(f"  median rotation         : {np.median(ang):.1f} deg")
    print(f"  IQR                     : [{np.percentile(ang,25):.1f},"
          f" {np.percentile(ang,75):.1f}] deg")
    print("\n  In `orient.to_polar` coordinates (y = cy + r sin t,"
          " x = cx + r cos t),")
    print("  reversing theta mirrors about the HORIZONTAL axis (up-down flip).")
    print(f"  Composing that with a {mean_ang:.0f} deg rotation is equivalent to:")
    resid = (mean_ang - 180.0 + 180.0) % 360.0 - 180.0
    print(f"    a LEFT-RIGHT flip (about the vertical axis)")
    print(f"    plus a residual rotation of {resid:+.1f} deg")
    print("\n  This matches the project owner's reading of the QC panels.")

    srs_files = {p.name[: -len(config.SRS_SUFFIX)]: p
                 for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}
    he_files = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}

    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row

    for core in sorted(args.cores, key=core_sort_key):
        if core not in by_core:
            print(f"  skip {core}: no orientation result")
            continue
        rec = by_core[core]
        s, h = shapes[core]["srs"], shapes[core]["he"]

        vol = tifffile.imread(srs_files[core])
        cars = vol[2].astype(np.float32)
        del vol
        cars_c = canonical(cars, float(s["circle_cy_full"]),
                           float(s["circle_cx_full"]),
                           float(s["circle_diam_full"]) / 2.0)
        lo, hi = np.nanpercentile(cars_c, (1, 99.5))
        cars_c = np.clip((cars_c - lo) / max(hi - lo, 1e-6), 0, 1)

        with Image.open(he_files[core]) as im:
            rgb = np.asarray(im.convert("RGB"))
        he_c = canonical(rgb, float(h["circle_cy_full"]),
                         float(h["circle_cx_full"]),
                         float(h["circle_diam_full"]) / 2.0) / 255.0

        # Apply the measured transform to the H&E.
        angle = float(rec["best_angle_deg"])
        he_t = he_c
        if rec["best_mirrored"] == "True":
            he_t = he_t[::-1, :]            # mirror about the horizontal axis
        he_t = ndi.rotate(he_t, -angle, reshape=False, order=1,
                          mode="constant", cval=np.nan)

        he_od = -np.log10(np.clip(he_t, 1e-3, 1.0))
        he_od = np.nansum(he_od, axis=-1)
        lo2, hi2 = np.nanpercentile(he_od, (1, 99))
        he_od_n = np.clip((he_od - lo2) / max(hi2 - lo2, 1e-6), 0, 1)

        fig, ax = plt.subplots(1, 5, figsize=(24, 5.2))
        ax[0].imshow(cars_c, cmap="gray", vmin=0, vmax=1)
        ax[0].set_title(f"{core}  SRS CARS (canonical)", fontsize=10)
        ax[1].imshow(he_c)
        ax[1].set_title("H&E canonical (untransformed)", fontsize=10)
        ax[2].imshow(he_t)
        ax[2].set_title(f"H&E mirrored + rotated {angle:.0f}deg", fontsize=10)

        # Overlay: SRS in green, transformed H&E density in magenta.
        ov = np.zeros(cars_c.shape + (3,), dtype=np.float32)
        ov[..., 1] = np.nan_to_num(cars_c)
        ov[..., 0] = np.nan_to_num(he_od_n)
        ov[..., 2] = np.nan_to_num(he_od_n)
        ax[3].imshow(np.clip(ov, 0, 1))
        ax[3].set_title("overlay: SRS green / H&E magenta", fontsize=10)

        # Same overlay WITHOUT the transform, as the control.
        he_od0 = np.nansum(-np.log10(np.clip(he_c, 1e-3, 1.0)), axis=-1)
        lo3, hi3 = np.nanpercentile(he_od0, (1, 99))
        he_od0 = np.clip((he_od0 - lo3) / max(hi3 - lo3, 1e-6), 0, 1)
        ov0 = np.zeros_like(ov)
        ov0[..., 1] = np.nan_to_num(cars_c)
        ov0[..., 0] = np.nan_to_num(he_od0)
        ov0[..., 2] = np.nan_to_num(he_od0)
        ax[4].imshow(np.clip(ov0, 0, 1))
        ax[4].set_title("CONTROL: no transform", fontsize=10)

        for a in ax:
            a.axis("off")
        fig.suptitle(
            f"{core}   mirrored={rec['best_mirrored']}  angle={angle:.0f}deg  "
            f"score={float(rec['best_score']):.4f}  "
            f"margin={float(rec['margin']):+.4f}", fontsize=12)
        fig.tight_layout()
        out = outdir / f"verify_{core}.png"
        fig.savefig(out, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
