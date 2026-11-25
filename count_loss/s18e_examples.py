"""
s18e -- example image figures from the s18 smoke inputs (vhe-detectors env).
Screening material only, not paper figures. Never overwrites.

  examples_models.png     rows = patches (2 per density class), columns =
                          real, real after stain round trip (R1), and the
                          four models' fakes (seed 42, epoch 30); StarDist
                          and CellViT-SAM-H counts under each image
  examples_detectors.png  central 50 um crop of a dense and a sparse patch,
                          real and fakes; rows = detectors, outlines in yellow
"""

import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
IN_DIR = __import__("vhe_paths").work("s18_metric_screen", "smoke")
OUT = os.path.join(_script_dir, "outputs", "s18_metric_screen", "smoke")
MIN_AREA = 10
PX_PER_UM = 512 / 100.0
IMGS = [("real_rgb", "Real H&E"), ("rt_rgb", "Real, round trip"), ("fake_rgb_CG", "CG"),
        ("fake_rgb_CGM", "CG+M"), ("fake_rgb_CGMA", "CG+M+A"), ("fake_rgb_CGMC", "CG+M+C")]
DETS = [("stardist", "StarDist (ref.)"), ("cellvit_sam", "CellViT-SAM-H"), ("hovernet", "HoVer-Net"),
        ("cellpose", "Cellpose-SAM"), ("instanseg", "InstanSeg")]


def count(lab):
    a = np.bincount(lab.ravel())[1:]
    return int((a >= MIN_AREA).sum())


def labels(det, uid, im):
    if det == "stardist":
        return np.load(os.path.join(IN_DIR, f"{uid}.npz"))["sd_lab_" + im]
    return np.load(os.path.join(IN_DIR, "det", f"{det}__{uid}.npz"))[im]


def scalebar(ax, um, size_px, color="white"):
    L = um * PX_PER_UM
    ax.plot([size_px - 12 - L, size_px - 12], [size_px - 14] * 2, color=color, lw=3, solid_capstyle="butt")
    ax.text(size_px - 12 - L / 2, size_px - 22, f"{um} um", color=color, ha="center", va="bottom", fontsize=7)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    for f in ("examples_models.png", "examples_detectors.png"):
        if os.path.exists(os.path.join(OUT, f)):
            sys.exit(f"{f} exists; refusing to overwrite")
    pt = pd.read_csv(os.path.join(IN_DIR, "patches.csv")).sort_values("n_real_cache")
    # two per density class: lower and upper part of each class
    picks = []
    for b in ("low", "med", "high"):
        d = pt[pt.bucket == b].reset_index(drop=True)
        picks += [d.iloc[len(d) // 4], d.iloc[(3 * len(d)) // 4]]
    picks = pd.DataFrame(picks)

    # ---- figure 1: models ----
    fig, ax = plt.subplots(len(picks), len(IMGS), figsize=(2.15 * len(IMGS), 2.45 * len(picks)))
    for i, r in enumerate(picks.itertuples()):
        z = np.load(os.path.join(IN_DIR, f"{r.patch_uid}.npz"))
        for j, (im, title) in enumerate(IMGS):
            a = ax[i, j]
            a.imshow(z[im]); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            nsd = count(labels("stardist", r.patch_uid, im)); ncv = count(labels("cellvit_sam", r.patch_uid, im))
            a.set_xlabel(f"StarDist {nsd}   CellViT {ncv}", fontsize=7.5, labelpad=2)
            if i == 0:
                a.set_title(title, fontsize=9, weight="bold")
            if j == 0:
                a.set_ylabel(f"{r.core}  ({r.bucket} density)", fontsize=8)
                scalebar(a, 20, 512)
    fig.suptitle("Real H&E vs the four models (seed 42, epoch 30); 100 um patches, held-out cores. "
                 "Counts: nuclei >= 10 px.", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(os.path.join(OUT, "examples_models.png"), dpi=130); plt.close(fig)

    # ---- figure 2: detectors on a central 50 um crop ----
    sel = [picks.iloc[-1].patch_uid, picks.iloc[0].patch_uid]  # dense, sparse
    cols = [(u, im, t) for u in sel for im, t in (IMGS[0], IMGS[2], IMGS[5])]
    c0, c1 = 128, 384
    fig, ax = plt.subplots(len(DETS), len(cols), figsize=(2.3 * len(cols), 2.4 * len(DETS)))
    for j, (u, im, t) in enumerate(cols):
        z = np.load(os.path.join(IN_DIR, f"{u}.npz"))
        for i, (d, dn) in enumerate(DETS):
            lab = labels(d, u, im)
            rgb = z[im].copy()
            rgb[find_boundaries(lab, mode="inner")] = (255, 225, 0)
            a = ax[i, j]
            a.imshow(rgb[c0:c1, c0:c1]); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            a.set_xlabel(f"{count(lab)} nuclei (whole patch)", fontsize=7.5, labelpad=2)
            if i == 0:
                a.set_title(f"{u.split('_eval')[0]}: {t}", fontsize=9, weight="bold")
            if j == 0:
                a.set_ylabel(dn, fontsize=9)
                scalebar(a, 10, c1 - c0)
    fig.suptitle("Detector outlines on the central 50 um of a dense (left) and a sparse (right) patch; "
                 "fakes from seed 42, epoch 30", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(os.path.join(OUT, "examples_detectors.png"), dpi=130); plt.close(fig)
    print("wrote", OUT, "patches:", list(picks.patch_uid))
