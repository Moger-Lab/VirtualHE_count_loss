"""
Supplementary Figure S7 (table) -- nuclear size (M1) and stain balance (M5)
by density class: fake / real, mean over 3 seeds [95% core-cluster CI],
epoch 30, real round-trip baseline. Descriptive only (these metrics fail the
seed-noise rule and are never used to rank models). Data:
data\\descriptive_by_class.csv (a02).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import pandas as pd
import matplotlib.pyplot as plt

LAB = {("M1 nuclear area", "stardist"): "nuclear area (StarDist)",
       ("M1 nuclear area", "cellvit_sam"): "nuclear area (CellViT-SAM-H)",
       ("M5 mean H (tissue)", "stardist"): "mean H in tissue",
       ("M5 mean E (tissue)", "stardist"): "mean E in tissue",
       ("M5 H fraction", "stardist"): "H fraction H/(H+E)"}
CLS = {"low": "sparse", "med": "intermediate", "high": "dense", "stromal": "stromal"}

if __name__ == "__main__":
    F.apply()
    T = pd.read_csv(os.path.join(F.DATA, "descriptive_by_class.csv"))
    rows, keys = [], []
    for (m, d), lab in LAB.items():
        for c in CLS:
            s = T[(T.metric == m) & (T.detector == d) & (T.subset == c)]
            if s.empty:
                continue
            rows.append([lab if c == "low" else "", f"{CLS[c]} ({int(s.n_patches.iloc[0])})"] +
                        [f"{s[s.model == k].value.iloc[0]:.2f} [{s[s.model == k].ci_lo.iloc[0]:.2f}, {s[s.model == k].ci_hi.iloc[0]:.2f}]"
                         for k in F.KEYS])
            keys.append((lab, c))
    fig, ax = plt.subplots(figsize=(F.DOUBLE, (len(rows) * 4.6 + 14) * F.MM))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["measure (fake / real)", "class (patches)"] + F.KEYS, loc="upper center",
                 cellLoc="center", colLoc="center", colWidths=[0.24, 0.15] + [0.152] * 4)
    t.auto_set_font_size(False); t.set_fontsize(6)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(F.GRID); c.set_linewidth(0.4)
        if i == 0:
            c.set_text_props(color=F.INK, fontweight="bold")
        if j == 0:
            c.set_text_props(ha="left"); c._loc = "left"
    t.scale(1, 1.25)
    F.save(fig, "figS7_tables", {"table": T})
