"""
Figure 1 (DRAFT schematic, to be refined by hand) -- study design.

(a) acquisition: TMA core, four label-free channels (only SRS is the model
    input) and the adjacent-section H&E (core B-1, owner's choice; rule-based
    alternatives in candidates\\fig1a_*).
(b) data: registration of the adjacent section, 100 um patch pairing,
    calibrated stain deconvolution of H&E into H and E channels.
(c) model: generator SRS -> (H, E), CycleGAN with two generators and two
    discriminators; the four models as a branch: CG -> CG+M (paired
    sampling and macro loss) -> CG+M+A (area loss) or CG+M+C (count loss).
(d) count loss: frozen counter on the generated H, 2 x 2 grid of 50 um
    cells per patch, compared with real counts; evaluation = StarDist on
    the rendered RGB output.
Pure vector except the real image thumbnails in (a) and (b).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F
import countlib as cl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

CORE = os.environ.get("FIG1_CORE", "B-1")  # owner's choice (2026-10-02): intact core, good mix of regions
IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
PATCH = "F-3_eval_3087_882"  # a Fig 2 candidate patch (intermediate), any patch works for the schematic


def box(ax, x, y, w, h, text, fc="#f0efec", ec=F.AXIS, fs=6, color=F.INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.015", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.15,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=F.INK2, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=0.7, mutation_scale=6, linestyle=ls,
                                shrinkA=0, shrinkB=0))


def thumb(ax, im, x, y, w, cmap=None):
    h = w * im.shape[0] / im.shape[1]
    ax.imshow(im, cmap=cmap, extent=(x, x + w, y, y + h), interpolation="lanczos", zorder=2)
    return h


def core_images():
    srs, he = cl.core_arrays(CORE)
    valid = np.asarray(np.load(os.path.join(cl.ALIGNED_DIR, f"{CORE}_valid.npy"), mmap_mode="r")[::8, ::8])
    s = np.asarray(srs[:, ::8, ::8]).astype(np.float32)
    out = []
    for c in range(4):
        lo, hi = np.percentile(s[c][valid], [0.5, 99.5])
        g = np.clip((s[c] - lo) / max(hi - lo, 1e-6), 0, 1); g[~valid] = 0
        out.append(g)
    h = np.asarray(he[::8, ::8]).copy(); h[~valid] = 255
    return out, h


if __name__ == "__main__":
    F.apply()
    from PIL import Image
    fig = plt.figure(figsize=(F.DOUBLE, 120 * F.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1.74); ax.set_ylim(0, 1.2); ax.axis("off"); ax.set_aspect("equal")

    # (a) acquisition
    ax.text(0.01, 1.17, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(0.05, 1.165, "acquisition: one TMA core, two adjacent sections", fontsize=6.5, color=F.INK2, va="top")
    ch, he = core_images()
    names = ["SHG", "SRS", "CARS", "TPEF"]
    for i, (g, n) in enumerate(zip(ch, names)):
        x = 0.05 + (i % 2) * 0.15; y = 0.975 - (i // 2) * 0.18
        thumb(ax, g, x, y - 0.0, 0.14, cmap="gray")
        ax.text(x + 0.07, y - 0.012, n + (" (input)" if n == "SRS" else ""), ha="center", va="top", fontsize=5.8,
                color=F.INK if n == "SRS" else F.MUTED, fontweight="bold" if n == "SRS" else "normal")
    ax.add_patch(Rectangle((0.198, 0.976), 0.144, 0.144, fill=False, ec=F.MODEL["CG+M+C"]["color"], lw=1.0, zorder=3))
    thumb(ax, he, 0.37, 0.89, 0.2)
    ax.text(0.47, 0.875, "adjacent section, H&E\n(no cell-level correspondence)", ha="center", va="top", fontsize=5.8,
            color=F.INK2)

    # (b) data
    ax.text(0.62, 1.17, "b", fontsize=9, fontweight="bold", va="top")
    ax.text(0.66, 1.165, "pairing and stain channels", fontsize=6.5, color=F.INK2, va="top")
    box(ax, 0.64, 1.0, 0.3, 0.12, "register adjacent section\nscale 1.02317, mirror,\nrigid refinement (S1, S2)")
    arrow(ax, 0.94, 1.06, 0.99, 1.06)
    box(ax, 0.99, 1.0, 0.26, 0.12, "100 um patches\n(441 px -> 512 px)\nsame location")
    arrow(ax, 1.25, 1.06, 1.30, 1.06)
    real = np.asarray(Image.open(os.path.join(IMG, "real", f"{PATCH}.png")).convert("RGB"))
    thumb(ax, real, 1.31, 1.0, 0.12)
    import countnet as cn
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    hch = cn.v1_common.rgb_to_he(real, calib)
    thumb(ax, 1 - (hch[0] + 1) / 2, 1.45, 1.0, 0.12, cmap="gray")
    thumb(ax, 1 - (hch[1] + 1) / 2, 1.59, 1.0, 0.12, cmap="gray")
    for x, t in ((1.37, "RGB"), (1.51, "H"), (1.65, "E")):
        ax.text(x, 0.985, t, ha="center", va="top", fontsize=5.8, color=F.INK2)
    ax.text(1.51, 0.955, "calibrated stain deconvolution", ha="center", va="top", fontsize=5.8, color=F.MUTED)

    # (c) model and the four models
    ax.text(0.62, 0.86, "c", fontsize=9, fontweight="bold", va="top")
    ax.text(0.66, 0.855, "unpaired translation (CycleGAN) and the four models", fontsize=6.5, color=F.INK2, va="top")
    box(ax, 0.64, 0.66, 0.14, 0.12, "SRS\npatch")
    arrow(ax, 0.78, 0.72, 0.83, 0.72)
    box(ax, 0.83, 0.66, 0.16, 0.12, "generator\nSRS -> (H, E)", fc="#e6eef9", ec=F.MODEL["CG+M+C"]["color"])
    arrow(ax, 0.99, 0.72, 1.04, 0.72)
    box(ax, 1.04, 0.66, 0.14, 0.12, "fake\nH, E")
    arrow(ax, 1.11, 0.66, 1.11, 0.61)
    box(ax, 1.03, 0.53, 0.16, 0.08, "discriminator\n(real H, E?)", fs=5.6)
    arrow(ax, 1.18, 0.72, 1.23, 0.72)
    box(ax, 1.23, 0.66, 0.16, 0.12, "generator\n(H, E) -> SRS")
    ax.plot([1.31, 1.31, 0.71], [0.66, 0.50, 0.50], color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    arrow(ax, 0.71, 0.50, 0.71, 0.655, color=F.MUTED)
    ax.text(1.0, 0.485, "cycle consistency: reconstructed SRS vs input\n(+ reverse cycle and second discriminator)",
            ha="center", va="top", fontsize=5.6, color=F.MUTED)
    # branch
    bx = 1.43
    def node(x, y, k, sub):
        st = F.MODEL[k]
        ax.plot([x], [y], marker=st["marker"], ms=6, color=st["color"], mec="white", mew=0.5, zorder=4)
        ax.text(x + 0.02, y, f"{k}\n{sub}", va="center", fontsize=5.6, color=F.INK, linespacing=1.1)
    node(bx, 0.80, "CG", "adversarial + cycle")
    node(bx, 0.68, "CG+M", "+ paired sampling, macro loss")
    node(bx + 0.02, 0.56, "CG+M+A", "+ area loss")
    node(bx + 0.02, 0.46, "CG+M+C", "+ count loss")
    ax.plot([bx, bx], [0.79, 0.69], color=F.INK2, lw=0.7)
    ax.plot([bx, bx, bx + 0.02], [0.67, 0.56, 0.56], color=F.INK2, lw=0.7)
    ax.plot([bx, bx, bx + 0.02], [0.60, 0.46, 0.46], color=F.INK2, lw=0.7)

    # (d) count loss
    ax.text(0.01, 0.6, "d", fontsize=9, fontweight="bold", va="top")
    ax.text(0.05, 0.595, "count loss (CG+M+C) and evaluation", fontsize=6.5, color=F.INK2, va="top")
    fake = np.asarray(Image.open(os.path.join(IMG, "A5_full", f"{PATCH}.png")).convert("RGB"))
    fh = cn.v1_common.rgb_to_he(fake, calib)[0]
    thumb(ax, 1 - (fh + 1) / 2, 0.05, 0.36, 0.16, cmap="gray")
    for k in (1,):
        ax.plot([0.05 + 0.08, 0.05 + 0.08], [0.36, 0.52], color=F.MODEL["CG+M+C"]["color"], lw=0.8, zorder=3)
        ax.plot([0.05, 0.21], [0.44, 0.44], color=F.MODEL["CG+M+C"]["color"], lw=0.8, zorder=3)
    ax.text(0.13, 0.345, "generated H, 2 x 2 cells of 50 um", ha="center", va="top", fontsize=5.6, color=F.INK2)
    arrow(ax, 0.22, 0.44, 0.27, 0.44)
    box(ax, 0.27, 0.39, 0.15, 0.1, "frozen counter\n(U-Net, val r 0.99)", fc="#e6eef9", ec=F.MODEL["CG+M+C"]["color"], fs=5.6)
    arrow(ax, 0.42, 0.44, 0.47, 0.44)
    box(ax, 0.47, 0.39, 0.13, 0.1, "4 cell counts\nvs real counts", fs=5.6)
    ax.text(0.47, 0.37, "local only: no patch-total term", ha="center", va="top", fontsize=5.6, color=F.MUTED)
    # evaluation strip
    box(ax, 0.05, 0.08, 1.0, 0.16, "evaluation, identical for every model: fake H, E rendered to RGB; StarDist nuclei per 100 um patch\n"
        "vs the adjacent real section (same round trip); 17 held-out cores, 1038 patches; 3 seeds x epochs 27-30\n"
        "per-patch r, log-log slope (1 = proportional), dense / sparse fake-to-real ratio (disc)",
        fc="white", fs=5.8)
    F.save(fig, "fig1_schematic_DRAFT", {})
