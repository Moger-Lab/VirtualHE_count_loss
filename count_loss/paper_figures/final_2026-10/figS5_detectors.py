"""
Supplementary Figure S5 -- detector robustness of the density metrics.

(a) M6 r, log-log slope, disc for the four models under StarDist (reference),
    CellViT-SAM-H and InstanSeg: seed values and mean with 95% core-cluster CI.
(b) paired differences (by seed) under each detector; CG+M+C minus CG+M+A is
    the pre-registered contrast (StarDist); the others are exploratory.
(c) round trip (rgb_to_he / he_to_rgb) on real patches: median rt / original
    nucleus count and area per detector (all 1038 real patches).
Epochs 27-30 mean, real round-trip baseline. Second detectors are
descriptive: they fail the pre-registered seed-noise rule (D-32).
Data: data\\detector_m6_*.csv (a04), outputs\\s18_metric_screen\\stageA\\R1_roundtrip_metrics.csv.
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

DETS = ["StarDist", "CellViT-SAM-H", "InstanSeg"]
DFILL = {"StarDist": "full", "CellViT-SAM-H": "open", "InstanSeg": "half"}
DOFF = {"StarDist": -0.22, "CellViT-SAM-H": 0.0, "InstanSeg": 0.22}
METRICS = [("r", "per-patch r", (-0.3, 1.0)), ("slope", "log-log slope", (-0.1, 1.1)), ("disc", "disc", (0, 1.1))]
STEPS = [("CG+M minus CG", "CG+M - CG"), ("CG+M+A minus CG+M", "CG+M+A - CG+M"),
         ("CG+M+C minus CG+M", "CG+M+C - CG+M"), ("CG+M+C minus CG+M+A", "CG+M+C - CG+M+A")]


def mark(ax, x, y, k, det, s=18):
    st = F.MODEL[k]
    fc = {"full": st["color"], "open": "white", "half": "none"}[DFILL[det]]
    kw = dict(marker=st["marker"], s=s, zorder=4, edgecolor=st["color"] if DFILL[det] != "full" else "white",
              linewidth=0.8 if DFILL[det] != "full" else 0.4, facecolor=fc)
    if DFILL[det] == "half":
        ax.scatter([x], [y], marker=st["marker"], s=s, color=st["color"], alpha=0.45, lw=0, zorder=4)
        kw["facecolor"] = "none"
    ax.scatter([x], [y], **kw)


def panel_a(fig, spec, T):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.35)
    axes = []
    for j, (m, lab, yl) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j]); axes.append(ax)
        if m != "r":
            ax.axhline(1, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
        for i, k in enumerate(F.KEYS):
            for det in DETS:
                r = T[(T.detector == det) & (T.model == k) & (T.metric == m)].iloc[0]
                x = i + DOFF[det]
                ax.plot([x, x], [r.ci_lo, r.ci_hi], color=F.MODEL[k]["color"], lw=1.0, solid_capstyle="butt", zorder=3)
                mark(ax, x, r.value, k, det)
        ax.set_xticks(range(4)); ax.set_xticklabels(F.KEYS, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylim(*yl); ax.set_ylabel(lab); ax.grid(axis="x", visible=False); ax.set_xlim(-0.5, 3.5)
    h = [Line2D([], [], color=F.INK2, marker="o", ls="", ms=4, label="StarDist"),
         Line2D([], [], color=F.INK2, marker="o", ls="", ms=4, markerfacecolor="white", label="CellViT-SAM-H"),
         Line2D([], [], color=F.INK2, marker="o", ls="", ms=4, markerfacecolor="#c3c2b7", label="InstanSeg")]
    axes[0].legend(handles=h, loc="upper left", fontsize=5.8, handletextpad=0.2, borderpad=0.2)
    F.panel_label(axes[0], "a", x=-0.38, y=1.05)


def panel_b(fig, spec, P):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.12, width_ratios=[1.5, 1, 1])
    first = None
    for j, (m, _, _) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j]); first = first or ax
        ax.axvline(0, color=F.MUTED, lw=0.6)
        for i, (name, short) in enumerate(STEPS):
            y0 = len(STEPS) - 1 - i
            pre = P[(P.comparison == name)].pre_registered.iloc[0]
            if pre:
                ax.axhspan(y0 - 0.45, y0 + 0.45, color="#f0efec", lw=0, zorder=0)
            for det in DETS:
                r = P[(P.detector == det) & (P.comparison == name) & (P.metric == m)].iloc[0]
                y = y0 - DOFF[det] * 1.2
                ax.plot([r.ci_lo, r.ci_hi], [y, y], color=F.MODEL[r.a]["color"], lw=1.0, solid_capstyle="butt", zorder=3)
                mark(ax, r["diff"], y, r.a, det, s=14)
        ax.set_yticks(range(len(STEPS)))
        if j == 0:
            labs = [f"{s}\n" + ("pre-registered" if P[P.comparison == n].pre_registered.iloc[0] else "exploratory")
                    for n, s in STEPS[::-1]]
            ax.set_yticklabels(labs, fontsize=6, color=F.INK2)
            ax.get_yticklabels()[0].set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(STEPS) - 0.4); ax.grid(axis="y", visible=False); ax.set_xlabel(f"difference in {m}")
    F.panel_label(first, "b", x=-0.75, y=1.04)


def panel_c(ax, R1):
    R1 = R1[R1.detector.isin(["stardist", "cellvit_sam", "instanseg"])].copy()
    R1["name"] = R1.detector.map({"stardist": "StarDist", "cellvit_sam": "CellViT-SAM-H", "instanseg": "InstanSeg"})
    ax.axis("off")
    rows = [[r.name, f"{r.n:.3f}", f"{r.med_area_um2:.3f}"] for r in R1.itertuples()]
    t = ax.table(cellText=rows, colLabels=["detector", "count\nrt / orig", "area\nrt / orig"], loc="center",
                 cellLoc="center", colLoc="center", colWidths=[0.52, 0.26, 0.26])
    t.auto_set_font_size(False); t.set_fontsize(6)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(F.GRID); c.set_linewidth(0.4)
        if i == 0:
            c.set_text_props(color=F.INK2); c.set_height(c.get_height() * 1.6)
    t.scale(1, 1.35)
    ax.text(0.0, 0.86, "round trip on real patches\n(median ratio, 1038 patches)", transform=ax.transAxes,
            fontsize=6.5, color=F.INK2, va="bottom")
    F.panel_label(ax, "c", x=-0.02, y=1.04)
    return R1[["name", "n", "med_area_um2"]].rename(columns={"n": "count_rt_over_orig", "med_area_um2": "area_rt_over_orig"})


if __name__ == "__main__":
    F.apply()
    T = pd.read_csv(os.path.join(F.DATA, "detector_m6_table.csv"))
    P = pd.read_csv(os.path.join(F.DATA, "detector_m6_paired.csv"))
    R1 = pd.read_csv(os.path.join(F.PROJECT, "outputs", "s18_metric_screen", "stageA", "R1_roundtrip_metrics.csv"))
    fig = plt.figure(figsize=(F.DOUBLE, 140 * F.MM))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05], hspace=0.62, wspace=0.45)
    panel_a(fig, gs[0, :], T)
    panel_b(fig, gs[1, 0:2], P)
    axc = fig.add_subplot(gs[1, 2]); sc = panel_c(axc, R1)
    F.save(fig, "figS5_detectors", {"a_m6": T, "b_paired": P, "c_roundtrip": sc})
