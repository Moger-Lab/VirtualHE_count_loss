"""Stage 1 QC - render what the core segmentation actually caught.

HANDOVER.md section 8: "Verify, don't assume." The first segmentation pass gave
a 13% scale CV across four cores, which cannot be right for objects that are
the same physical punch. Numbers alone cannot say whether that is a
segmentation failure or a data fact, so this renders the masks over the images.

Produces one PNG per requested core: SRS channel composite with mask outline,
and H&E with mask outline, side by side with the shape numbers.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/03_segmentation_qc.py --cores A-1 A-2 A-5 A-6
"""

from __future__ import annotations

import argparse
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
from skimage import measure
from skimage.transform import resize

import config
from src import coreseg

Image.MAX_IMAGE_PIXELS = None

DS_VIEW = 8


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def srs_composite(vol: np.ndarray, ds: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """False-colour composite plus the individual normalised channels."""
    if vol.shape[0] != 4 and vol.shape[-1] == 4:
        vol = np.moveaxis(vol, -1, 0)
    c, y, x = vol.shape
    chans = []
    for i in range(c):
        s = resize(vol[i].astype(np.float32), (y // ds, x // ds),
                   order=1, anti_aliasing=True, preserve_range=True)
        lo, hi = np.percentile(s, (1.0, 99.5))
        chans.append(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1))
    # SHG -> blue-ish, fsSRS -> green, CARS -> red, TPF folded into blue
    rgb = np.stack([chans[2], chans[1], np.maximum(chans[0], chans[3])], axis=-1)
    return np.clip(rgb, 0, 1), chans


def outline(ax, mask: np.ndarray, colour: str, lw: float = 1.4):
    for c in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], colour, linewidth=lw)


def draw_circle(ax, s, ds: int, shape, colour: str = "white"):
    """Overlay the fitted punch circle, in downsampled display coordinates.

    Axis limits are pinned to the image extent afterwards: a circle larger than
    the frame would otherwise autoscale the axes and shrink the image to an
    unreadable thumbnail, hiding the very failure being inspected.
    """
    h, w = shape[:2]
    if np.isfinite(s.circle_diam_full):
        t = np.linspace(0, 2 * np.pi, 400)
        r = s.circle_diam_full / 2.0 / ds
        cy, cx = s.circle_cy_full / ds, s.circle_cx_full / ds
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), colour,
                linewidth=1.2, linestyle="--", alpha=0.9)
        ax.plot([cx], [cy], marker="+", color=colour, markersize=9)
    else:
        ax.text(0.5, 0.04, "circle fit REJECTED", transform=ax.transAxes,
                ha="center", color="red", fontsize=11, weight="bold")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", nargs="+", required=True)
    ap.add_argument("--tag", default="qc")
    args = ap.parse_args()

    config.ensure_project_dirs()
    outdir = config.FIGURES_DIR / "02_core_segmentation"
    outdir.mkdir(parents=True, exist_ok=True)

    srs_files = {p.name[: -len(config.SRS_SUFFIX)]: p
                 for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}
    he_files = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}

    for core in sorted(args.cores, key=core_sort_key):
        vol = tifffile.imread(srs_files[core])
        rgb_srs, chans = srs_composite(vol, DS_VIEW)
        srs_raw, srs_thr, srs_bg = coreseg.segment_srs(vol, ds=DS_VIEW)
        srs_mask = coreseg.footprint(srs_raw)
        s_srs = coreseg.describe(srs_raw, core, "srs", DS_VIEW, srs_thr, srs_bg)
        del vol

        with Image.open(he_files[core]) as im:
            he_full = np.asarray(im.convert("RGB"))
        hy, hx = he_full.shape[:2]
        he_small = resize(he_full.astype(np.float32),
                          (hy // DS_VIEW, hx // DS_VIEW, 3),
                          order=1, anti_aliasing=True, preserve_range=True) / 255.0
        he_raw, he_thr, he_bg, he_blank = coreseg.segment_he(he_full, ds=DS_VIEW)
        he_mask = coreseg.footprint(he_raw)
        s_he = coreseg.describe(he_raw, core, "he", DS_VIEW, he_thr, he_bg,
                                exclude=he_blank)
        del he_full

        fig, axes = plt.subplots(2, 4, figsize=(19, 10))
        fig.suptitle(
            f"{core}   |   SRS circle {s_srs.circle_diam_full:.0f}px "
            f"rms {s_srs.circle_rms_rel:.4f}  frag {s_srs.n_fragments}  "
            f"clip {s_srs.border_frac:.3f}"
            f"   ||   H&E circle {s_he.circle_diam_full:.0f}px "
            f"rms {s_he.circle_rms_rel:.4f}  frag {s_he.n_fragments}  "
            f"clip {s_he.border_frac:.3f}"
            f"   ||   scale(circle) "
            f"{s_he.circle_diam_full / s_srs.circle_diam_full:.4f}"
            f"   scale(area) "
            f"{s_he.equiv_diam_full / s_srs.equiv_diam_full:.4f}",
            fontsize=12)

        names = ["ch0 SHG", "ch1 fsSRS", "ch2 CARS", "ch3 TPF"]
        for i, (ch, nm) in enumerate(zip(chans, names)):
            axes[0, i].imshow(ch, cmap="magma", vmin=0, vmax=1)
            axes[0, i].set_title(nm, fontsize=10)
            axes[0, i].axis("off")

        axes[1, 0].imshow(rgb_srs)
        axes[1, 0].set_title("SRS composite", fontsize=10)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(rgb_srs)
        outline(axes[1, 1], srs_mask, "cyan")
        draw_circle(axes[1, 1], s_srs, DS_VIEW, rgb_srs.shape, "white")
        axes[1, 1].set_title("SRS + footprint + fitted punch", fontsize=10)
        axes[1, 1].axis("off")

        axes[1, 2].imshow(he_small)
        axes[1, 2].set_title("H&E", fontsize=10)
        axes[1, 2].axis("off")

        axes[1, 3].imshow(he_small)
        outline(axes[1, 3], he_mask, "lime")
        draw_circle(axes[1, 3], s_he, DS_VIEW, he_small.shape, "black")
        axes[1, 3].set_title("H&E + footprint + fitted punch", fontsize=10)
        axes[1, 3].axis("off")

        fig.tight_layout()
        out = outdir / f"{args.tag}_{core}.png"
        fig.savefig(out, dpi=85, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

