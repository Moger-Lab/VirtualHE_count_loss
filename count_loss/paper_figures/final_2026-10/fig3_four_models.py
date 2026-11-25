"""
Figure 3 -- the four models on the held-out evaluation cores (17 cores,
1038 patches), StarDist, epochs 27-30 mean, real round-trip baseline.

(a) r, log-log slope and disc per model: three seed values (small marks),
    mean over seeds with 95% core-cluster bootstrap CI.
(b) paired differences (by seed) with 95% CI. CG+M+C minus CG+M+A is the
    pre-registered contrast; the others are exploratory.
(c) median fake/real per real-count bin (seed-mean per patch), 95%
    core-cluster bootstrap band; 1 = proportional.
Data: data\\four_model_rt_*.csv (a01_tables.py).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import NullFormatter

METRICS = [("r", "per-patch r", (-0.25, 1.0)),
           ("slope", "log-log slope", (-0.1, 1.1)),
           ("disc", "disc (dense / sparse ratio)", (0.0, 1.1))]
XPOS = {"CG": 0.0, "CG+M": 1.0, "CG+M+A": 2.25, "CG+M+C": 3.25}
STEPS = [  # comparison name, short label
    ("CG+M minus CG", "CG+M - CG"),
    ("CG+M+A minus CG+M", "CG+M+A - CG+M"),
    ("CG+M+C minus CG+M", "CG+M+C - CG+M"),
    ("CG+M+C minus CG+M+A", "CG+M+C - CG+M+A"),
]
BINS = np.array([0, 15, 25, 35, 45, 55, 70, 90, 115, 250])


def panel_a(fig, spec, T, PR):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.45)
    rows, axes = [], []
    for j, (m, lab, yl) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j]); axes.append(ax)
        if m != "r":
            ax.axhline(1.0, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
            ax.text(-0.4, 1.015, "proportional", fontsize=5.5, color=F.MUTED, ha="left", va="bottom")
        if m != "disc":
            ax.axhline(0.0, color=F.AXIS, lw=0.6)
        for k in F.KEYS:
            st, x = F.MODEL[k], XPOS[k]
            t = T.loc[k]
            v = PR[PR.model == k][m].values
            ax.scatter(x + np.array([-0.16, 0.0, 0.16]), v, s=9, marker=st["marker"], facecolor="none",
                       edgecolor=st["color"], linewidth=0.7, zorder=3)
            ax.plot([x + 0.32] * 2, [t[m + "_lo"], t[m + "_hi"]], color=st["color"], lw=1.4, solid_capstyle="butt", zorder=4)
            ax.scatter([x + 0.32], [t[m]], s=24, marker=st["marker"], color=st["color"], edgecolor="white",
                       linewidth=0.5, zorder=5)
            for s, val in zip(F.SEEDS, v):
                rows.append(dict(metric=m, model=k, kind=f"seed {s}", value=val))
            rows.append(dict(metric=m, model=k, kind="mean", value=t[m], ci_lo=t[m + "_lo"], ci_hi=t[m + "_hi"]))
        ax.set_xticks([XPOS[k] + 0.08 for k in F.KEYS])
        ax.set_xticklabels(F.KEYS, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.45, 3.85); ax.set_ylim(*yl); ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
        # branch bracket: CG+M+A and CG+M+C are alternatives from CG+M
        y0 = 0.86
        ax.plot([XPOS["CG+M+A"] - 0.18, XPOS["CG+M+A"] - 0.18, XPOS["CG+M+C"] + 0.36, XPOS["CG+M+C"] + 0.36],
                [y0 - 0.03, y0, y0, y0 - 0.03], color=F.MUTED, lw=0.5)
        ax.text((XPOS["CG+M+A"] + XPOS["CG+M+C"] + 0.32) / 2, y0 + 0.01 * (yl[1] - yl[0]), "alternatives",
                fontsize=5.5, color=F.MUTED, ha="center", va="bottom")
    F.panel_label(axes[0], "a", x=-0.42, y=1.06)
    return pd.DataFrame(rows)


def panel_b(fig, spec, P):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.12, width_ratios=[1.6, 1, 1])
    rows, first = [], None
    for j, (m, lab, _) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j]); first = first or ax
        ax.axvline(0, color=F.MUTED, lw=0.6)
        for i, (name, short) in enumerate(STEPS):
            r = P[(P.comparison == name) & (P.metric == m)].iloc[0]
            st = F.MODEL[r.a]; y = len(STEPS) - 1 - i
            if r.pre_registered:
                ax.axhspan(y - 0.42, y + 0.42, color="#f0efec", lw=0, zorder=0)
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=st["color"], lw=1.4, solid_capstyle="butt", zorder=3)
            ax.scatter([r["diff"]], [y], s=24, marker=st["marker"], color=st["color"], edgecolor="white",
                       linewidth=0.5, zorder=4)
            rows.append(dict(comparison=short, metric=m, pre_registered=bool(r.pre_registered), diff=r["diff"],
                             ci_lo=r.ci_lo, ci_hi=r.ci_hi, excludes_zero=bool(r.excl_0)))
        ax.set_yticks(range(len(STEPS)))
        if j == 0:
            labs = []
            for name, short in STEPS[::-1]:
                pre = P[(P.comparison == name)].pre_registered.iloc[0]
                labs.append(f"{short}\n" + ("pre-registered" if pre else "exploratory"))
            ax.set_yticklabels(labs, fontsize=6, color=F.INK2)
            for tl, (name, _) in zip(ax.get_yticklabels(), STEPS[::-1]):
                if P[(P.comparison == name)].pre_registered.iloc[0]:
                    tl.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(STEPS) - 0.4); ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"difference in {m}")
    F.panel_label(first, "b", x=-0.78, y=1.04)
    return pd.DataFrame(rows)


def binned(nf, nr, cores, boot_idx):
    ratio = nf / np.maximum(nr, 1)
    b = np.digitize(nr, BINS) - 1
    rows = []
    for k in np.unique(b):
        sel = b == k
        if sel.sum() < 8:
            continue
        bs = []
        for idx in boot_idx:
            s = idx[b[idx] == k]
            if len(s):
                bs.append(np.median(ratio[s]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(bin_lo=BINS[k], bin_hi=BINS[k + 1], x=float(np.median(nr[sel])), n=int(sel.sum()),
                         median=float(np.median(ratio[sel])), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def panel_c(ax, pt, R):
    nr = pt.n_real_rt.values
    bidx = F.boot_indices(pt, n=500)
    ax.axhline(1.0, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.text(165, 1.04, "proportional", fontsize=5.5, color=F.MUTED, ha="right", va="bottom")
    rows = []
    for k in F.KEYS:
        st = F.MODEL[k]
        d = binned(F.seed_mean_fake(R, k), nr, pt.core.values, bidx)
        ax.fill_between(d.x, d.lo, d.hi, color=st["color"], alpha=0.12, lw=0)
        ax.plot(d.x, d["median"], color=st["color"], marker=st["marker"], ms=3.2, markeredgecolor="white",
                markeredgewidth=0.4, label=k, zorder=3)
        rows.append(d.assign(model=k))
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([10, 20, 50, 100]); ax.set_xticklabels(["10", "20", "50", "100"]); ax.set_xlim(8, 170)
    ax.set_xlabel("real nuclei per 100 um patch"); ax.set_ylabel("median fake / real (log scale)")
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"]); ax.set_ylim(0.25, 12)
    ax.legend(handles=F.legend_handles(), loc="upper right", ncol=2, handletextpad=0.3, columnspacing=0.8,
              borderpad=0.2, labelspacing=0.3)
    F.panel_label(ax, "c", x=-0.24, y=1.04)
    return pd.concat(rows)


if __name__ == "__main__":
    F.apply()
    T = pd.read_csv(os.path.join(F.DATA, "four_model_rt_table.csv")).set_index("model")
    PR = pd.read_csv(os.path.join(F.DATA, "four_model_rt_per_run.csv"))
    P = pd.read_csv(os.path.join(F.DATA, "four_model_rt_paired.csv"))
    pt, R = F.load_all()
    import example_panels as EP
    ex = EP.has_images()
    TOP = 120.0
    H = TOP + 104.0 if ex else 128.0
    fig = plt.figure(figsize=(F.DOUBLE, H * F.MM))
    lim = dict(top=1 - 4.0 / H, bottom=1 - (TOP - 14.0) / H) if ex else {}
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.95], hspace=0.66, wspace=0.5, **lim)
    sa = panel_a(fig, gs[0, :], T, PR)
    sb = panel_b(fig, gs[1, 0:2], P)
    axc = fig.add_subplot(gs[1, 2]); sc = panel_c(axc, pt, R)
    src = {"a_metrics": sa, "b_paired": sb, "c_binned": sc}
    if ex:
        V = pd.read_csv(os.path.join(F.DATA, "fig2_final_selection.csv"))
        src["d_examples"] = EP.outline_rows(fig, H, 9.0, TOP + 3.0, V, size=22.5, gap=3.0)
    else:
        print("example images not found: panel d skipped")
    F.save(fig, "fig3_four_models", src)
