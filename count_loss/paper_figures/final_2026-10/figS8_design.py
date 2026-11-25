"""
Supplementary Figure S8 (was Figure 6; moved to the supplement by the owner,
2026-10-02, D-39) -- why a local count loss (design rationale).

(a) CG+M+A, the area-loss model: fake/real H-positive area (the quantity the
    area loss acts on; generator H channel at the calibrated nuclear
    threshold) and fake/real StarDist count, by density class. Median of
    per-patch ratios, mean over 3 seeds, epochs 27-30 mean; 95% core-cluster
    bootstrap CI. Held-out evaluation cores.
(b) share of a count loss carried by the patch-total term vs the per-cell
    term, at 12.5 um and 50 um cells, for a 40% undercount (analytic, from
    the real cell-count distribution; numbers as in the A3 design analysis).
(c) supervision reliability between the SRS section and the adjacent H&E
    section against cell size: SRS-H&E architecture correlation, mean over
    75 registered cores (v2 registration scale sweep). The two loss cell
    sizes are marked.
Data: data\\area_signal_per_patch.csv (a03), paper_figures\\out\\
fig4_global_vs_local_d_loss_shares_source.csv (read-only), v2 06_registration.json.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F
import countlib as cl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CL = [("low", "sparse"), ("med", "interm."), ("high", "dense")]
AREA_C, COUNT_C = "#898781", F.MODEL["CG+M+A"]["color"]


def panel_a(ax, pt, R, A):
    tags = F.MODEL["CG+M+A"]["tags"]
    nr = pt.n_real_rt.values
    area = {t: A[f"hpos_{t}"].values / A.hpos_real.values for t in tags}
    cnt = {t: R[t].n_fake.values / np.maximum(nr, 1) for t in tags}
    b = pt.bucket.values
    bidx = F.boot_indices(pt)
    rows = []
    for j, (bk, lab) in enumerate(CL):
        for name, D, dx, mk, col in (("H-positive area", area, -0.12, "s", F.INK2), ("nucleus count", cnt, 0.12, "D", COUNT_C)):
            m = b == bk
            v = np.mean([np.median(D[t][m]) for t in tags])
            bs = [np.mean([np.median(D[t][i][m[i]]) for t in tags]) for i in bidx]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ax.plot([j + dx] * 2, [lo, hi], color=col, lw=1.2, solid_capstyle="butt")
            ax.scatter([j + dx], [v], marker=mk, s=24, color=col, edgecolor="white", lw=0.5, zorder=3,
                       label=name if j == 0 else None)
            rows.append(dict(bucket=bk, quantity=name, value=v, ci_lo=lo, ci_hi=hi, n=int(m.sum())))
    S = pd.DataFrame(rows)
    for name, dx, col in (("H-positive area", -0.12, F.INK2), ("nucleus count", 0.12, COUNT_C)):
        s = S[S.quantity == name]
        ax.plot(np.arange(3) + dx, s.value, color=col, lw=0.7, zorder=2)
    ax.axhline(1, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in CL]); ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.45, 1.45); ax.set_ylabel("fake / real (CG+M+A)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", handletextpad=0.3, borderpad=0.2, fontsize=6)
    ax.set_title("area-loss signal vs count deficit", loc="left", color=F.INK2)
    F.panel_label(ax, "a", x=-0.3, y=1.06)
    return S


def panel_b(ax, L):
    L = L.sort_values("cell_um")
    x = np.arange(len(L))
    tot = L.total_share.values
    ax.bar(x, tot, width=0.55, color=F.INK2, label="patch-total term")
    ax.bar(x, 1 - tot, bottom=tot, width=0.55, color="#c3c2b7", label="per-cell term")
    for i, t in enumerate(tot):
        ax.text(i, t / 2, f"{t:.0%}", ha="center", va="center", fontsize=6.5, color="white")
        ax.text(i, t + (1 - t) / 2, f"{1 - t:.0%}", ha="center", va="center", fontsize=6.5, color=F.INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{c:g} um cells\n({m:.1f} nuclei / cell)" for c, m in zip(L.cell_um, L.mean_real_per_cell)],
                                         fontsize=6)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_ylabel("share of count loss (40% undercount)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=5.8, handlelength=1.0, columnspacing=0.8, borderpad=0.1)
    ax.set_title("small cells: the patch total dominates", loc="left", color=F.INK2)
    F.panel_label(ax, "b", x=-0.3, y=1.06)
    return L


def panel_c(ax, J):
    S = pd.DataFrame(J["scale_sweep"])
    ax.plot(S.um, S.r_mean, color=F.INK2, marker="o", ms=3.5, lw=1.0)
    for um, lab, dy in ((12.5, "12.5 um cells", -0.075), (50, "50 um cells\n(CG+M+C)", 0.035)):
        r = float(np.interp(um, S.um, S.r_mean))
        ax.scatter([um], [r], s=40, facecolor="none", edgecolor=F.MODEL["CG+M+C"]["color"] if um == 50 else F.MUTED,
                   lw=1.0, zorder=4)
        ax.text(um * (1.06 if dy < 0 else 1), r + dy, lab, ha="left" if dy < 0 else "center", va="bottom",
                fontsize=5.8, color=F.INK2)
    ax.set_xscale("log"); ax.set_xticks([6, 12.5, 25, 50, 100]); ax.set_xticklabels(["6", "12.5", "25", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(0, 0.6); ax.set_xlabel("cell size (um)"); ax.set_ylabel("SRS vs adjacent H&E\narchitecture r (mean, 75 cores)")
    ax.set_title("adjacent-section agreement grows with scale", loc="left", color=F.INK2)
    F.panel_label(ax, "c", x=-0.3, y=1.06)
    return S


if __name__ == "__main__":
    F.apply()
    pt, R = F.load_all()
    A = pd.read_csv(os.path.join(F.DATA, "area_signal_per_patch.csv"))
    assert list(A.patch_uid) == list(pt.patch_uid)
    L = pd.read_csv(os.path.join(F.PROJECT, "paper_figures", "out", "fig4_global_vs_local_d_loss_shares_source.csv"))
    J = json.load(open(os.path.join(cl.V2_RESULTS, "06_registration.json")))
    fig, axs = plt.subplots(1, 3, figsize=(F.DOUBLE, 62 * F.MM), gridspec_kw=dict(wspace=0.55))
    sa = panel_a(axs[0], pt, R, A)
    sb = panel_b(axs[1], L)
    sc = panel_c(axs[2], J)
    print(sa.round(3).to_string(index=False))
    F.save(fig, "figS8_design", {"a_area_vs_count": sa, "b_loss_share": sb, "c_reliability": sc})
