"""
Supplementary Figure S2 -- registration quality of the adjacent sections
(v2 pipeline outputs, read-only: VirtualH&E_Analysis_v2\\results).

(a) residual offset after registration per core (um), by registration tier.
(b) SRS-H&E architecture agreement at 50 um per core (matched sections)
    against the mismatched-core null (mean, +/- 1 SD band, 95th percentile).
(c) cohort facts: scale, mirroring, exclusions, tier counts.
Tiers are ordinal (registered > weak signal > misaligned): one-hue ink ramp
plus marker shape.
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

TIER = {"A_registered": ("registered", "#3b3a38", "o"), "B_weak_signal": ("weak", "#898781", "s"),
        "C_misaligned": ("misaligned", "#c3c2b7", "D")}


def strip(ax, T, col):
    rng = np.random.default_rng(0)
    for i, (t, (lab, c, mk)) in enumerate(TIER.items()):
        v = T[T.tier == t][col].values
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=11, marker=mk, color=c, edgecolor="white", lw=0.3,
                   zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=F.INK, lw=1.0, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{v[0]}\n(n = {(T.tier == k).sum()})" for k, v in TIER.items()])
    ax.set_xlim(-0.6, 2.6); ax.grid(axis="x", visible=False)


if __name__ == "__main__":
    F.apply()
    T = pd.read_csv(os.path.join(cl.V2_RESULTS, "16_transforms.csv"))
    J = json.load(open(os.path.join(cl.V2_RESULTS, "06_registration.json")))
    fig = plt.figure(figsize=(F.DOUBLE, 62 * F.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    strip(ax, T, "resid_um")
    ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_ylabel("residual offset (um, log scale)")
    ax.text(0.98, 0.03, f"median, all 75 cores:\n{T.resid_um.median():.1f} um", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color=F.INK2)
    F.panel_label(ax, "a", x=-0.3, y=1.04)

    ax = fig.add_subplot(gs[0, 1])
    mu, sd, p95 = J["null_mean"], J["null_sd"], J["null_p95"]
    ax.axhspan(mu - sd, mu + sd, color="#f0efec", lw=0, zorder=0)
    ax.axhline(mu, color=F.MUTED, lw=0.6)
    ax.axhline(p95, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    strip(ax, T, "r_50um")
    ax.text(2.55, p95 + 0.015, "null 95th pct.", fontsize=5.5, color=F.MUTED, ha="right", va="bottom")
    ax.text(-0.55, mu - sd + 0.015, "null mean +/- 1 SD\n(mismatched cores)", fontsize=5.5, color=F.MUTED, ha="left", va="bottom")
    ax.set_ylabel("architecture r at 50 um (SRS vs H&E)")
    ax.text(0.98, 0.97, f"matched mean {J['matched_mean']:.2f}\n= {J['separation_null_sd']:.1f} SD above null",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color=F.INK2)
    ax.set_ylim(-0.45, 1.0)
    F.panel_label(ax, "b", x=-0.3, y=1.04)

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    lines = [("paired cores", "77"),
             ("excluded", "A-1 (damaged H&E)\nA-6 (saturated SRS)"),
             ("registered", "75"),
             ("cohort scale", f"{J['cohort_scale']:.5f} H&E px / SRS px"),
             ("H&E flip", f"mirrored left-right ({int(T.mirror_he_lr.sum())}/75)"),
             ("tiers", f"{(T.tier == 'A_registered').sum()} registered, {(T.tier == 'B_weak_signal').sum()} weak,"
                       f"\n{(T.tier == 'C_misaligned').sum()} misaligned"),
             ("pixel size", "0.1953 um / px (render)")]
    y = 0.98
    for k, v in lines:
        ax.text(0.0, y, k, transform=ax.transAxes, fontsize=6, color=F.INK2, va="top")
        ax.text(0.40, y, v, transform=ax.transAxes, fontsize=6, color=F.INK, va="top", linespacing=1.15)
        y -= 0.135 + 0.06 * v.count("\n")
    F.panel_label(ax, "c", x=-0.05, y=1.04)
    src = T[["core", "tier", "resid_um", "r_50um", "mirror_he_lr", "scale_he_px_per_srs_px", "eval_eligible"]].assign(
        null_mean=mu, null_sd=sd, null_p95=p95, matched_mean=J["matched_mean"], separation_sd=J["separation_null_sd"])
    F.save(fig, "figS2_registration", {"cores": src})
