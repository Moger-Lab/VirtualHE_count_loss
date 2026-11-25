"""
Supplementary Figure S4 -- every run and every core.

(a) fake vs real nuclei per patch, log-log, one panel per run (4 models x
    3 seeds; epochs 27-30 mean; real = round trip); per-run r and slope;
    y = x dashed.
(b) per core: median fake / real per model (mean over seeds; small marks =
    seeds), cores ordered by median real count.
Held-out evaluation cores (17 cores, 1038 patches), StarDist.
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

TICKS = [10, 30, 100]


if __name__ == "__main__":
    F.apply()
    pt, R = F.load_all()
    PR = pd.read_csv(os.path.join(F.DATA, "four_model_rt_per_run.csv")).set_index("run")
    nr = pt.n_real_rt.values
    fig = plt.figure(figsize=(F.DOUBLE, 200 * F.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.3, 1], hspace=0.22)
    ga = GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[0], wspace=0.1, hspace=0.12)
    rows = []
    for j, k in enumerate(F.KEYS):
        for i, (t, s) in enumerate(zip(F.MODEL[k]["tags"], F.SEEDS)):
            ax = fig.add_subplot(ga[i, j])
            nf = R[t].n_fake.values
            ok = (nr > 0) & (nf > 0)
            ax.scatter(nr[ok], nf[ok], s=2.5, color=F.MODEL[k]["color"], alpha=0.45, lw=0, rasterized=True)
            ax.plot([4, 260], [4, 260], color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
            b, a = np.polyfit(np.log(nr[ok]), np.log(nf[ok]), 1)
            xs = np.array([6, 220]); ax.plot(xs, np.exp(a) * xs ** b, color=F.INK, lw=0.9)
            ax.set_xscale("log"); ax.set_yscale("log")
            for ax_ in (ax.xaxis, ax.yaxis):
                ax_.set_minor_formatter(NullFormatter())
            ax.set_xticks(TICKS); ax.set_yticks(TICKS)
            ax.set_xticklabels([str(v) for v in TICKS] if i == 2 else [])
            ax.set_yticklabels([str(v) for v in TICKS] if j == 0 else [])
            ax.set_xlim(4, 260); ax.set_ylim(4, 260); ax.set_aspect("equal")
            ax.text(0.05, 0.95, f"r {PR.loc[t, 'r']:.2f}\nslope {PR.loc[t, 'slope']:.2f}", transform=ax.transAxes,
                    va="top", fontsize=5.8, color=F.INK2, linespacing=1.1)
            if i == 0:
                ax.set_title(k, fontsize=7, color=F.INK)
            if j == 0:
                ax.set_ylabel(f"seed {s}\nfake nuclei")
            if i == 2:
                ax.set_xlabel("real nuclei")
            if i == 0 and j == 0:
                F.panel_label(ax, "a", x=-0.55, y=1.1)
            for u, c, x, y in zip(pt.patch_uid, pt.core, nr, nf):
                rows.append(dict(model=k, run=t, seed=s, patch_uid=u, core=c, n_real_rt=x, n_fake=y))

    ax = fig.add_subplot(gs[1])
    order = pt.groupby("core").n_real_rt.median().sort_values()
    cores = list(order.index)
    crow = []
    for k in F.KEYS:
        st = F.MODEL[k]
        per_seed = []
        for t in st["tags"]:
            ratio = R[t].n_fake.values / np.maximum(nr, 1)
            per_seed.append(pd.Series(ratio, index=pt.core.values).groupby(level=0).median().reindex(cores).values)
        per_seed = np.array(per_seed)
        x = np.arange(len(cores)) + (F.KEYS.index(k) - 1.5) * 0.17
        for sv in per_seed:
            ax.scatter(x, sv, s=4, marker=st["marker"], facecolor="none", edgecolor=st["color"], lw=0.5, alpha=0.6)
        ax.scatter(x, per_seed.mean(0), s=14, marker=st["marker"], color=st["color"], edgecolor="white", lw=0.4, zorder=3)
        for c, m_, sv in zip(cores, per_seed.mean(0), per_seed.T):
            crow.append(dict(core=c, model=k, median_ratio_seed_mean=m_, seeds=";".join(f"{v:.3f}" for v in sv)))
    ax.axhline(1, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xticks(range(len(cores)))
    ax.set_xticklabels([f"{c}\n{order[c]:.0f}" for c in cores], fontsize=5.8)
    ax.set_xlabel("core (median real nuclei per patch)"); ax.set_ylabel("median fake / real")
    ax.grid(axis="x", visible=False); ax.set_xlim(-0.6, len(cores) - 0.4)
    ax.legend(handles=F.legend_handles(), loc="upper right", ncol=4, handletextpad=0.3, columnspacing=1.0, borderpad=0.2)
    F.panel_label(ax, "b", x=-0.07, y=1.04)
    F.save(fig, "figS4_seeds_cores", {"a_per_run": pd.DataFrame(rows), "b_per_core": pd.DataFrame(crow)})
