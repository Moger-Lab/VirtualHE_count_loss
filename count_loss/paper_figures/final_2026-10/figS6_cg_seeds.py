"""
Supplementary Figure S6 -- CG is unstable across seeds.

(a) per-seed r, slope, disc for all four models (epochs 27-30, rt
    baseline): the three CG seeds spread far more than any other model's.
    Built once: figS6a_seed_values.
(b) example outputs of the three CG seeds (epoch 30) on the same patches,
    with SRS input and adjacent real H&E. Candidate versions A-D and the
    unselected E use the same rule-selected patch sets as Fig 2
    (data\\fig2_candidates.csv); contact sheet in candidates\\.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F
import fig2_candidates as G2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLS_S6 = [("srs", "SRS input", None), ("real", "adjacent real H&E", "n_real_orig"),
           ("CG_full", "CG seed 42", "n_fake_CG"), ("CG_full_s7", "CG seed 7", "n_fake_CG_s7"),
           ("CG_full_s123", "CG seed 123", "n_fake_CG_s123")]


def seed_values():
    PR = pd.read_csv(os.path.join(F.DATA, "four_model_rt_per_run.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(F.DOUBLE * 0.8, 50 * F.MM), gridspec_kw=dict(wspace=0.45))
    for ax, (m, lab) in zip(axs, (("r", "per-patch r"), ("slope", "log-log slope"), ("disc", "disc"))):
        for i, k in enumerate(F.KEYS):
            v = PR[PR.model == k]
            for s, val in zip(v.seed, v[m]):
                ax.scatter([i], [val], marker=F.MODEL[k]["marker"], s=16, color=F.MODEL[k]["color"], edgecolor="white", lw=0.4)
                if k == "CG":
                    ax.text(i + 0.14, val, str(s), fontsize=5.5, color=F.INK2, va="center")
            ax.plot([i - 0.2, i + 0.2], [v[m].mean()] * 2, color=F.INK, lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels(F.KEYS, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel(lab); ax.grid(axis="x", visible=False); ax.set_xlim(-0.5, 3.6)
        if m != "disc":
            ax.axhline(0, color=F.AXIS, lw=0.6)
    F.panel_label(axs[0], "a", x=-0.45, y=1.04)
    sd = PR.groupby("model")[["r", "slope", "disc"]].std().reindex(F.KEYS)
    print("seed SD by model:\n", sd.round(3))
    F.save(fig, "figS6a_seed_values", {"per_run": PR, "seed_sd": sd.reset_index()})


if __name__ == "__main__":
    F.apply()
    seed_values()
    G2.COLS = COLS_S6
    C = pd.read_csv(os.path.join(F.DATA, "fig2_candidates.csv"))
    pngs = []
    for v in ["A", "B", "C", "D", "E (unselected)"]:
        V = C[C.version == v]
        fig = G2.build(V, f"Figure S6b candidate {v} (CG, three seeds, epoch 30)")
        F.save(fig, f"figS6b_candidate_{v[0]}", {"patches": V}, sub=G2.SUB)
        pngs.append((v, os.path.join(F.OUT, G2.SUB, f"figS6b_candidate_{v[0]}.png")))
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for _, p in pngs]
    ims = [i.resize((i.width // 2, i.height // 2), Image.LANCZOS) for i in ims]
    W = sum(i.width for i in ims) + 40 * (len(ims) + 1); H = max(i.height for i in ims) + 90
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arialbd.ttf", 44)
    x = 40
    for (v, _), im in zip(pngs, ims):
        d.text((x, 20), v, fill=(11, 11, 11), font=font); sheet.paste(im, (x, 80)); x += im.width + 40
    sheet.save(os.path.join(F.OUT, G2.SUB, "figS6b_contact_sheet.png"))
    print("wrote contact sheet")
