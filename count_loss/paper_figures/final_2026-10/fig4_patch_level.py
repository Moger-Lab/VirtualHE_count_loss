"""
Figure 4 -- patch- and core-level view of density conditioning (held-out
evaluation cores, 17 cores, 1038 patches; StarDist; per-patch fake count =
mean over epochs 27-30 and over the 3 seeds; real = round-trip StarDist).

(a) log-log scatter, fake vs real count per patch, one panel per model;
    points coloured by their core's median real count (one-hue ramp, so a
    core reads as a cluster without 17 categorical colours); y = x dashed;
    solid line = mean of the three per-seed log-log fits.
(b) core medians, fake vs real; arrows from CG+M+A to CG+M+C per core.
(c) per-patch fake/real by density bucket (buckets on the original-RGB real
    count, as everywhere), violins per model, median marked.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F
import core_panels as CP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

TICKS = [5, 10, 20, 50, 100, 200]
LIM = (4, 260)
CORE_CMAP = colors.LinearSegmentedColormap.from_list("cores", [cm.Greens(0.30), cm.Greens(0.95)])
LABEL_CORES = ["H-2"]
HIGHLIGHT = None  # set to a core name in __main__ when panel d is drawn


def logaxes(ax, lim=LIM):
    ax.set_xscale("log"); ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_formatter(NullFormatter())
    ax.set_xticks(TICKS); ax.set_xticklabels([str(t) for t in TICKS])
    ax.set_yticks(TICKS); ax.set_yticklabels([str(t) for t in TICKS])
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.plot(lim, lim, color=F.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)


def per_seed_fit(R, k, nr):
    sl, ic = [], []
    for t in F.MODEL[k]["tags"]:
        nf = R[t].n_fake.values
        m = (nr > 0) & (nf > 0)
        b, a = np.polyfit(np.log(nr[m]), np.log(nf[m]), 1)
        sl.append(b); ic.append(a)
    return float(np.mean(sl)), float(np.mean(ic))


def panel_a(fig, spec, pt, R, T, core_med):
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.12, hspace=0.32)
    nr = pt.n_real_rt.values
    norm = colors.LogNorm(vmin=core_med.min(), vmax=core_med.max())
    cvals = CORE_CMAP(norm(pt.core.map(core_med).values))
    rows, axes = [], []
    for i, k in enumerate(F.KEYS):
        ax = fig.add_subplot(gs[i // 2, i % 2]); axes.append(ax)
        st = F.MODEL[k]
        nf = F.seed_mean_fake(R, k)
        ok = (nr > 0) & (nf > 0)
        ax.scatter(nr[ok], nf[ok], s=3.5, c=cvals[ok], alpha=0.75, lw=0, rasterized=True, zorder=2)
        if HIGHLIGHT:
            h = ok & (pt.core.values == HIGHLIGHT)
            ax.scatter(nr[h], nf[h], s=6, facecolor="none", edgecolor=F.INK, lw=0.35, zorder=3, rasterized=True)
        b, a = per_seed_fit(R, k, nr)
        xs = np.array([6, 220])
        ax.plot(xs, np.exp(a) * xs ** b, color=st["color"], lw=1.4, zorder=4)
        logaxes(ax)
        t = T.loc[k]
        ax.text(0.04, 0.96, f"{k}", transform=ax.transAxes, fontsize=7, fontweight="bold", va="top", color=F.INK)
        ax.plot([0.06], [0.80], transform=ax.transAxes, marker=st["marker"], color=st["color"], ms=4,
                markeredgecolor="white", markeredgewidth=0.4)
        ax.text(0.11, 0.80, f"slope {t['slope']:.2f}\nr {t['r']:.2f}", transform=ax.transAxes, fontsize=6,
                va="center", color=F.INK2, linespacing=1.1)
        if i % 2:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("fake nuclei")
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("real nuclei")
        for u, c, x, y, okk in zip(pt.patch_uid, pt.core, nr, nf, ok):
            rows.append(dict(model=k, patch_uid=u, core=c, n_real_rt=x, n_fake_seed_mean=y, plotted=bool(okk)))
        rows.append(dict(model=k, patch_uid="FIT", core="", n_real_rt=np.nan, n_fake_seed_mean=np.nan,
                         plotted=True, fit_slope=b, fit_intercept_log=a))
    # colour key for the core ramp
    cax = fig.add_axes([0, 0, 1, 1]); cax.set_visible(False)
    F.panel_label(axes[0], "a", x=-0.36, y=1.05)
    return pd.DataFrame(rows), axes, norm


def panel_b(ax, pt, R):
    nr = pt.n_real_rt.values
    d = pd.DataFrame({"core": pt.core.values, "real": nr,
                      "A": F.seed_mean_fake(R, "CG+M+A"), "C": F.seed_mean_fake(R, "CG+M+C")})
    g = d.groupby("core").median()
    logaxes(ax, (15, 140))
    sA, sC = F.MODEL["CG+M+A"], F.MODEL["CG+M+C"]
    for c, r in g.iterrows():
        ax.annotate("", xy=(r.real, r.C), xytext=(r.real, r.A),
                    arrowprops=dict(arrowstyle="-|>", color=F.INK2, lw=0.7, mutation_scale=5, shrinkA=2.5, shrinkB=3.5),
                    zorder=5)
    ax.scatter(g.real, g.A, marker=sA["marker"], s=16, color=sA["color"], edgecolor="white", lw=0.4, zorder=3)
    ax.scatter(g.real, g.C, marker=sC["marker"], s=18, color=sC["color"], edgecolor="white", lw=0.4, zorder=4)
    for c in LABEL_CORES:
        r = g.loc[c]
        ax.text(r.real * 1.07, r.C, c, fontsize=6, color=F.INK2, va="center")
    ax.set_xlabel("core median real nuclei"); ax.set_ylabel("core median fake nuclei")
    ax.legend(handles=F.legend_handles(["CG+M+A", "CG+M+C"]), loc="upper left", handletextpad=0.3, borderpad=0.2)
    ax.text(0.97, 0.04, "arrow: CG+M+A to CG+M+C\nper core (17)", transform=ax.transAxes, fontsize=5.5,
            color=F.MUTED, ha="right", va="bottom")
    ax.text(118, 126, "y = x", fontsize=5.5, color=F.MUTED, ha="right", va="bottom", rotation=45)
    F.panel_label(ax, "b", x=-0.3, y=1.03)
    return g.reset_index().rename(columns={"A": "CG+M+A_median_fake", "C": "CG+M+C_median_fake", "real": "median_real_rt"})


def panel_c(ax, pt, R):
    nr = pt.n_real_rt.values
    b = pt.bucket.values
    rows = []
    ax.axhline(0.0, color=F.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)  # log(1)
    w = 0.19
    for j, bk in enumerate(["low", "med", "high"]):
        for i, k in enumerate(F.KEYS):
            st = F.MODEL[k]
            ratio = (F.seed_mean_fake(R, k) / np.maximum(nr, 1))[b == bk]
            lr = np.log(np.clip(ratio, 0.05, None))
            x = j + (i - 1.5) * (w + 0.03)
            vp = ax.violinplot([lr], positions=[x], widths=w, showextrema=False, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(st["color"]); body.set_edgecolor(st["color"]); body.set_alpha(0.35); body.set_linewidth(0.5)
            med = np.median(ratio)
            ax.scatter([x], [np.log(med)], marker=st["marker"], s=14, color=st["color"], edgecolor="white", lw=0.4, zorder=4)
            q1, q3 = np.percentile(ratio, [25, 75])
            ax.plot([x, x], np.log([q1, q3]), color=st["color"], lw=1.0, zorder=3)
            rows.append(dict(bucket=bk, model=k, n=int(len(ratio)), median=med, q25=q1, q75=q3,
                             p05=np.percentile(ratio, 5), p95=np.percentile(ratio, 95)))
    yt = [0.25, 0.5, 1, 2, 4]
    ax.set_yticks(np.log(yt)); ax.set_yticklabels([str(v) for v in yt]); ax.set_ylim(np.log(0.15), np.log(6.5))
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([F.BUCKET_LABEL[k] for k in ["low", "med", "high"]])
    ax.set_xlabel("real density bucket (nuclei per patch)"); ax.set_ylabel("fake / real per patch (log scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=F.legend_handles(), loc="upper right", ncol=4, handletextpad=0.3, columnspacing=1.0, borderpad=0.2)
    F.panel_label(ax, "c", x=-0.075, y=1.07)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    F.apply()
    T = pd.read_csv(os.path.join(F.DATA, "four_model_rt_table.csv")).set_index("model")
    pt, R = F.load_all()
    core_med = pt.groupby("core").n_real_rt.median()
    has_d = os.path.isdir(CP.IMGC)  # the core image mosaics are not distributed with the code (see README)
    fig = plt.figure(figsize=(F.DOUBLE, (252 if has_d else 165) * F.MM))
    gs = GridSpec(3 if has_d else 2, 2, figure=fig, height_ratios=[1.45, 0.75, 1.75][:3 if has_d else 2],
                  width_ratios=[1.25, 1], hspace=0.32, wspace=0.28)
    HIGHLIGHT = "H-2" if has_d else None
    sa, axes_a, norm = panel_a(fig, gs[0, 0], pt, R, T, core_med)
    axb = fig.add_subplot(gs[0, 1]); sb = panel_b(axb, pt, R)
    axc = fig.add_subplot(gs[1, :]); sc = panel_c(axc, pt, R)
    src = {"a_scatter": sa, "b_core_medians": sb, "c_violins": sc}
    if has_d:
        cor, pcn = CP.build(fig, gs[2, :], ["H-2"], pt.groupby("core").n_real.median(), letter="d")
        src["d_cores"] = cor
        src["d_patch_counts"] = pcn
    else:
        print("img_cores not found: panel d skipped (panels a-c unaffected)")
    # core ramp key (for panel a) in the free space under panel b
    fig.canvas.draw()  # resolve equal-aspect axes positions first
    pb = axb.get_position(); pa = axes_a[2].get_position()
    kax = fig.add_axes([pb.x0, pa.y0 - 0.045, pb.width * 0.7, 0.008])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CORE_CMAP), cax=kax, orientation="horizontal")
    cb.set_ticks([30, 50, 100]); cb.set_ticklabels(["30", "50", "100"]); cb.minorticks_off()
    cb.solids.set_edgecolor("face")
    cb.outline.set_linewidth(0.4); kax.tick_params(labelsize=5.5, length=2)
    kax.set_title("panel a, point colour: core median real nuclei", fontsize=5.5, color=F.INK2, loc="left", pad=2)
    if HIGHLIGHT:
        axes_a[1].text(0.97, 0.05, f"ringed: {HIGHLIGHT} (panel d)", transform=axes_a[1].transAxes, fontsize=5.5, color=F.INK2, ha="right", va="bottom")
    F.save(fig, "fig4_patch_level", src)

