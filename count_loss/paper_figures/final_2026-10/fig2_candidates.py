"""
Figure 2 candidates -- example outputs. One figure per candidate version
(A-D, rule-based quality selection; E = unselected random counterpart for
the supplement) and one contact sheet with all versions side by side.

Rows: sparse, intermediate, dense, stromal (one patch each). Columns: SRS
input (channel 1), adjacent real H&E, CG, CG+M, CG+M+A, CG+M+C, all seed 42
epoch 30 (fixed rule; CG seed 42 is CG's weakest seed on r, stated in the
legend). Numbers = StarDist nuclei in the 100 um patch (real: original RGB;
fakes: the evaluation counts). Patch choice: data\\fig2_candidates.csv
(g01_candidate_images.py). Outputs: candidates\\.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
COLS = [("srs", "SRS input", None), ("real", "adjacent real H&E", "n_real_orig"),
        ("CG_full", "CG", "n_fake_CG"), ("CGM_full", "CG+M", "n_fake_CG+M"),
        ("A1_full", "CG+M+A", "n_fake_CG+M+A"), ("A5_full", "CG+M+C", "n_fake_CG+M+C")]
ROWS = ["sparse", "intermediate", "dense", "stromal"]
SUB = "candidates"


def read(set_, uid):
    from PIL import Image
    return np.asarray(Image.open(os.path.join(IMG, set_, f"{uid}.png")).convert("RGB"))


def build(V, title):
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(F.DOUBLE, 128 * F.MM),
                             gridspec_kw=dict(wspace=0.04, hspace=0.22))
    for i, row in enumerate(ROWS):
        r = V[V.row == row].iloc[0]
        for j, (set_, head, ncol) in enumerate(COLS):
            ax = axes[i, j]
            ax.imshow(read(set_, r.patch_uid), interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if ncol:
                ax.text(0.5, -0.035, f"{int(r[ncol])} nuclei", transform=ax.transAxes, ha="center", va="top",
                        fontsize=6, color=F.INK if set_ == "real" else F.INK2,
                        fontweight="bold" if set_ == "real" else "normal")
            if i == 0:
                ax.set_title(head, fontsize=6.5, color=F.INK, pad=4)
                if set_ not in ("srs", "real"):
                    k = head if head in F.MODEL else head.split()[0]
                    ax.add_line(Line2D([0.5], [1.19], transform=ax.transAxes, marker=F.MODEL[k]["marker"], ms=4,
                                       color=F.MODEL[k]["color"], markeredgecolor="white", markeredgewidth=0.4, clip_on=False))
            if j == 0:
                ax.text(-0.08, 0.5, f"{row}\n{r.patch_uid}", transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=5.8, color=F.INK2, linespacing=1.15)
        if i == len(ROWS) - 1:  # 20 um scale bar on the SRS image of the last row (512 px = 100 um)
            ax0 = axes[i, 0]
            ax0.plot([20, 20 + 512 * 0.2], [490, 490], color="white", lw=1.6, solid_capstyle="butt")
            ax0.text(20 + 51.2, 478, "20 um", color="white", fontsize=5.5, ha="center", va="bottom")
    if title:
        fig.suptitle(title, x=0.12, ha="left", fontsize=7, color=F.INK2, y=0.995)
    return fig


if __name__ == "__main__":
    F.apply()
    C = pd.read_csv(os.path.join(F.DATA, "fig2_candidates.csv"))
    pngs = []
    for v in ["A", "B", "C", "D", "E (unselected)"]:
        V = C[C.version == v]
        tag = v[0]
        fig = build(V, f"Figure 2 candidate {v}")
        F.save(fig, f"fig2_candidate_{tag}", {"patches": V}, sub=SUB)
        pngs.append((v, os.path.join(F.OUT, SUB, f"fig2_candidate_{tag}.png")))

    # contact sheet: all versions side by side, labelled
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for _, p in pngs]
    h = max(i.height for i in ims); scale = 0.5
    ims = [i.resize((int(i.width * scale), int(i.height * scale)), Image.LANCZOS) for i in ims]
    W = sum(i.width for i in ims) + 40 * (len(ims) + 1); H = max(i.height for i in ims) + 90
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arialbd.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    x = 40
    for (v, _), im in zip(pngs, ims):
        d.text((x, 20), v, fill=(11, 11, 11), font=font)
        sheet.paste(im, (x, 80)); x += im.width + 40
    p = os.path.join(F.OUT, SUB, "fig2_contact_sheet.png")
    if os.path.exists(p) and os.environ.get("FIG_REDRAFT") != "1":
        raise FileExistsError(p)
    sheet.save(p)
    print("wrote", p)
