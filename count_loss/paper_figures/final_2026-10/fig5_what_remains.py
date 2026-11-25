"""
Figure 5 -- what the models share (descriptive; never used to rank models).

(a) M1: median fake/real nuclear area by density class, StarDist (filled)
    and CellViT-SAM-H (open); mean over seeds with 95% core-cluster CI.
(b) M5 stain balance by density class and stromal patches: mean H in
    tissue, mean E in tissue, H fraction H/(H+E), fake/real.
(c) stromal fabrication against output level, every evaluated checkpoint
    (4 models x 3 seeds x epochs 27-30).
(a, b) epoch 30, real round-trip baseline. Data: data\\descriptive_*.csv,
checkpoint_level_fabrication.csv (a02_descriptive.py).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

CL_LAB = {"low": "sparse", "med": "interm.", "high": "dense", "stromal": "stromal"}
OFF = {k: (i - 1.5) * 0.17 for i, k in enumerate(F.KEYS)}


def dots(ax, D, classes, open_=False, dx=0.0):
    rows = []
    for j, cl in enumerate(classes):
        for k in F.KEYS:
            st = F.MODEL[k]
            r = D[(D.subset == cl) & (D.model == k)].iloc[0]
            x = j + OFF[k] + dx
            ax.plot([x, x], [r.ci_lo, r.ci_hi], color=st["color"], lw=1.0, solid_capstyle="butt", zorder=3,
                    alpha=0.6 if open_ else 1)
            ax.scatter([x], [r.value], s=16, marker=st["marker"], zorder=4,
                       facecolor="white" if open_ else st["color"], edgecolor=st["color"] if open_ else "white",
                       linewidth=0.8 if open_ else 0.4)
            rows.append(dict(subset=cl, model=k, value=r.value, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([CL_LAB[c] for c in classes])
    ax.set_xlim(-0.6, len(classes) - 0.4); ax.grid(axis="x", visible=False)
    ax.axhline(1.0, color=F.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    return pd.DataFrame(rows)


def panel_a(ax, T):
    s = []
    for det, op, dx in (("stardist", False, -0.04), ("cellvit_sam", True, 0.04)):
        D = T[(T.metric == "M1 nuclear area") & (T.detector == det)]
        s.append(dots(ax, D, ["low", "med", "high"], open_=op, dx=dx).assign(detector=det))
    ax.set_ylim(0.6, 1.15); ax.set_ylabel("fake / real median nuclear area")
    ax.set_title("nuclear size (M1)", loc="left", color=F.INK2)
    h = F.legend_handles() + [Line2D([], [], color=F.INK2, marker="o", ls="", ms=4, label="StarDist"),
                              Line2D([], [], color=F.INK2, marker="o", ls="", ms=4, markerfacecolor="white", label="CellViT-SAM-H")]
    ax.legend(handles=h, loc="lower right", ncol=3, handletextpad=0.2, columnspacing=0.7, borderpad=0.2, fontsize=5.8)
    ax.text(2.55, 1.005, "equal size", fontsize=5.5, color=F.MUTED, ha="right", va="bottom")
    F.panel_label(ax, "a", x=-0.16, y=1.06)
    return pd.concat(s)


def panel_b(fig, spec, T):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.3)
    rows, axes = [], []
    for j, (metric, lab) in enumerate((("M5 mean H (tissue)", "mean haematoxylin in tissue"),
                                       ("M5 mean E (tissue)", "mean eosin in tissue"),
                                       ("M5 H fraction", "H fraction, H / (H + E)"))):
        ax = fig.add_subplot(gs[0, j]); axes.append(ax)
        D = T[T.metric == metric]
        rows.append(dots(ax, D, ["low", "med", "high", "stromal"]).assign(metric=metric))
        ax.axvspan(2.5, 3.5, color="#f0efec", lw=0, zorder=0)
        ax.set_ylim(0.6, 1.9); ax.set_title(lab, loc="left", color=F.INK2)
        if j == 0:
            ax.set_ylabel("fake / real")
        else:
            ax.set_yticklabels([])
    F.panel_label(axes[0], "b", x=-0.3, y=1.08)
    return pd.concat(rows)


def panel_c(ax, C):
    for k in F.KEYS:
        st = F.MODEL[k]
        g = C[C.model == k]
        ax.scatter(g.level, g.stromal_fab, s=14, marker=st["marker"], color=st["color"], edgecolor="white",
                   linewidth=0.4, zorder=3, label=k)
    b, a = np.polyfit(C.level, C.stromal_fab, 1)
    xs = np.array([C.level.min() - 0.05, C.level.max() + 0.05])
    ax.plot(xs, a + b * xs, color=F.INK2, lw=0.8, ls=(0, (4, 2)), zorder=2)
    r = np.corrcoef(C.level, C.stromal_fab)[0, 1]
    wr = {k: np.corrcoef(g.level, g.stromal_fab)[0, 1] for k, g in C.groupby("model")}
    ax.text(0.04, 0.96, f"48 checkpoints, r = {r:.2f}\nwithin model r = {min(wr.values()):.2f} to {max(wr.values()):.2f}",
            transform=ax.transAxes, fontsize=6, color=F.INK2, va="top")
    ax.set_xlabel("output level (median fake / real)"); ax.set_ylabel("stromal fabrication\n(fake nuclei per empty 12.5 um cell)")
    ax.legend(handles=F.legend_handles(), loc="lower right", handletextpad=0.3, borderpad=0.2, labelspacing=0.3)
    F.panel_label(ax, "c", x=-0.3, y=1.06)
    return C.assign(fit_slope=b, fit_intercept=a, pooled_r=r)


if __name__ == "__main__":
    F.apply()
    T = pd.read_csv(os.path.join(F.DATA, "descriptive_by_class.csv"))
    C = pd.read_csv(os.path.join(F.DATA, "checkpoint_level_fabrication.csv"))
    import example_panels as EP
    ex = EP.has_images()
    TOP = 104.0
    H = TOP + 99.0 if ex else 125.0
    fig = plt.figure(figsize=(F.DOUBLE, H * F.MM))
    lim = dict(top=1 - 4.0 / H, bottom=1 - (TOP - 12.0) / H) if ex else {}
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1], hspace=0.55, wspace=0.42, **lim)
    axa = fig.add_subplot(gs[0, 0]); sa = panel_a(axa, T)
    axc = fig.add_subplot(gs[0, 1]); sc = panel_c(axc, C)
    sb = panel_b(fig, gs[1, :], T)
    src = {"a_area": sa, "b_stain": sb, "c_level_fabrication": sc}
    if ex:
        V = pd.read_csv(os.path.join(F.DATA, "fig2_final_selection.csv"))
        PT = EP.patch_table()
        src["d_size_examples"] = EP.size_block(fig, H, 10.0, TOP, V, PT, size=13.0, gap=2.2, letter="d")
        src["e_stain_examples"] = EP.stain_block(fig, H, 98.0, TOP, V, PT, size=13.0, gap=2.2, letter="e")
        src["f_fabrication_example"] = EP.empty_block(fig, H, 10.0, TOP + 50.0, V, size=24.0, gap=3.0, letter="f")
    else:
        print("example images not found: panels d-f skipped")
    F.save(fig, "fig5_what_remains", src)
