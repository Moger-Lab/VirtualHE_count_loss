"""
Whole-core panels shared by Fig 4d and Fig S10 (data: g03_core_full.py).

For each core, three rows of five images (real H&E, then the four models; seed 42,
epoch 30):
  row 1  the H&E as generated, no overlay, cut to the round core
  row 2  one red dot per StarDist nucleus on a light disc with a grey silhouette
         of the REAL tissue (the same in every column, so where a model puts, or
         omits, nuclei relative to the real tissue is visible)
  row 3  the evaluation patches (100 um squares) coloured by their nucleus count;
         one square = one point of Fig 4a, so the graph and the core can be read
         against each other
Every 100 um grid position touching the core is shown in rows 1-2; the evaluation
patches sit at their exact positions (their dots are the evaluation detections),
the rim patches are not in any evaluation number and are not in row 3. Patches
are generated independently, so faint tile seams are expected in the fakes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

IMGC = __import__("vhe_paths").work("paper_figs_2026-10", "img_cores")
SETS = [("real", "real H&E"), ("CG", "CG"), ("CG+M", "CG+M"), ("CG+M+A", "CG+M+A"), ("CG+M+C", "CG+M+C")]
RULE = {"densest": "densest core", "typical": "typical core", "candidate": "core", "training": "training core"}
DOT = "#d6191f"
DISC = "#f6f5f1"
TISSUE = (0.86, 0.85, 0.81, 1.0)
SHOW = 4    # image down-sampling for display (dots stay at full-grid coordinates)
GRID = 512  # grid px per 100 um patch
S = 512 / 441
RD = 0.985  # displayed disc = fitted disc x RD (trims the pale generated background at the rim)
HCMAP = colors.LinearSegmentedColormap.from_list("count", [cm.Greens(0.08), cm.Greens(0.97)])
HNORM = colors.Normalize(vmin=0, vmax=140)


def silhouette(core):
    """Real-tissue mask on the mosaic grid / 8 (smoothed summed OD > 0.20)."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
    im = Image.open(os.path.join(IMGC, f"{core}_real_full.png")).convert("RGB")
    w, h = im.size
    a = np.asarray(im.resize((w // 8, h // 8), Image.BILINEAR)).astype(np.float32)
    od = -np.log10((a + 1.0) / 256.0).sum(-1)
    m = gaussian_filter(od, 1.0) > 0.20
    return binary_closing(binary_opening(m, iterations=1), iterations=2)


def heat(ax, ct, tiles, cx, cy, r):
    """Evaluation patches coloured by nucleus count; returns the per-patch counts."""
    clip = Circle((cx, cy), RD * r, transform=ax.transData)
    vals = []
    for t in tiles.itertuples():
        n = int(ct.get(t.patch_uid, 0)); vals.append(n)
        rc = Rectangle((t.col * GRID, t.row * GRID), GRID, GRID, fc=HCMAP(HNORM(min(n, 140))), ec="white", lw=0.3, zorder=2)
        ax.add_patch(rc); rc.set_clip_path(clip)
    return np.array(vals)


def build(fig, spec, cores, core_med, letter=None, ddir=None):
    from PIL import Image
    ddir = ddir or F.DATA
    nuc_all = pd.read_csv(os.path.join(ddir, "fig4d_nuclei_full.csv"))
    nuc = nuc_all[nuc_all.in_disc]
    cor = pd.read_csv(os.path.join(ddir, "fig4d_cores_full.csv"))
    circ = pd.read_csv(os.path.join(ddir, "fig4d_circle.csv")).set_index("core")
    gs = GridSpecFromSubplotSpec(3 * len(cores), len(SETS), subplot_spec=spec, wspace=0.04, hspace=0.16)
    first = None
    counts, meds, patch_counts = {}, {}, []
    for ci, core in enumerate(cores):
        rule = cor[cor.core == core].rule.iloc[0]
        sil = silhouette(core)
        tiles = pd.read_csv(os.path.join(ddir, f"fig4d_tiles_{core}.csv"))
        tiles = tiles[tiles.in_eval]
        cx, cy, r = circ.loc[core, "cx_native"] * S, circ.loc[core, "cy_native"] * S, circ.loc[core, "r_native"] * S
        lo_x, hi_x, lo_y, hi_y = cx - r * 1.02, cx + r * 1.02, cy - r * 1.02, cy + r * 1.02
        for j, (s, head) in enumerate(SETS):
            im = Image.open(os.path.join(IMGC, f"{core}_{s}_full.png")).convert("RGB")
            w, h = im.size
            n = nuc[(nuc.set == s) & (nuc.core == core)]
            n = n[(n.gx - cx) ** 2 + (n.gy - cy) ** 2 <= (RD * r) ** 2]
            counts[(core, s)] = len(n)
            for rr in range(3):
                ax = fig.add_subplot(gs[3 * ci + rr, j])
                first = first or ax
                if rr == 0:
                    sm = np.asarray(im.resize((w // SHOW, h // SHOW), Image.LANCZOS)).copy()
                    yy, xx = np.mgrid[0:sm.shape[0], 0:sm.shape[1]]
                    sm[((xx + 0.5) * SHOW - cx) ** 2 + ((yy + 0.5) * SHOW - cy) ** 2 > (RD * r) ** 2] = 255
                    ax.imshow(sm, extent=(0, w, h, 0), interpolation="lanczos")
                elif rr == 1:
                    ax.add_patch(Circle((cx, cy), RD * r, fc=DISC, ec="none", zorder=0))
                    rgba = np.zeros(sil.shape + (4,), np.float32); rgba[sil] = TISSUE
                    yy, xx = np.mgrid[0:sil.shape[0], 0:sil.shape[1]]
                    rgba[((xx + 0.5) * 8 - cx) ** 2 + ((yy + 0.5) * 8 - cy) ** 2 > (RD * r) ** 2] = 0
                    ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest", zorder=1)
                    ax.scatter(n.gx, n.gy, s=0.6, color=DOT, lw=0, alpha=0.9, rasterized=True, zorder=2)
                    ax.text(0.5, -0.01, f"{counts[(core, s)]} nuclei", transform=ax.transAxes, ha="center", va="top", fontsize=6,
                            color=F.INK if s == "real" else F.INK2, fontweight="bold" if s == "real" else "normal")
                else:
                    ax.add_patch(Circle((cx, cy), RD * r, fc=DISC, ec="none", zorder=0))
                    ct = nuc_all[(nuc_all.set == s) & (nuc_all.core == core) & nuc_all.in_eval].groupby("patch_uid").size()
                    v = heat(ax, ct, tiles, cx, cy, r)
                    if len(v):
                        meds[(core, s)] = float(np.median(v))
                        patch_counts += [dict(core=core, set=s, patch_uid=u, n_nuclei=int(x)) for u, x in zip(tiles.patch_uid, v)]
                        ax.text(0.5, -0.01, f"median {meds[(core, s)]:.0f} per patch", transform=ax.transAxes, ha="center", va="top",
                                fontsize=6, color=F.INK if s == "real" else F.INK2, fontweight="bold" if s == "real" else "normal")
                    else:
                        ax.text(0.5, 0.5, "no evaluation\npatches" if j == 2 else "", transform=ax.transAxes, ha="center", va="center",
                                fontsize=6, color=F.MUTED)
                    if j == 0 and len(v):
                        cax = ax.inset_axes([0.05, -0.27, 0.9, 0.045])
                        cb = fig.colorbar(cm.ScalarMappable(norm=HNORM, cmap=HCMAP), cax=cax, orientation="horizontal")
                        cb.set_ticks([0, 50, 100, 140]); cb.minorticks_off(); cb.outline.set_linewidth(0.4)
                        cax.tick_params(labelsize=5.2, length=1.5, pad=1)
                        cax.set_xlabel("nuclei per patch (one square = one point in Fig 4a)", fontsize=5.2, color=F.INK2, labelpad=1.5, loc="left")
                ax.set_xlim(lo_x, hi_x); ax.set_ylim(hi_y, lo_y); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if ci == 0 and rr == 0:
                    ax.set_title(head, fontsize=6.5, color=F.INK, pad=4)
                    if s != "real":
                        ax.add_line(Line2D([0.5], [1.19], transform=ax.transAxes, marker=F.MODEL[s]["marker"], ms=4,
                                           color=F.MODEL[s]["color"], markeredgecolor="white", markeredgewidth=0.4, clip_on=False))
                if j == 0:
                    lab = ["H&E", "nuclei (StarDist)", "count per patch"][rr]
                    ax.text(-0.06, 0.5, lab, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=5.8, color=F.INK2)
                    if rr == 1:
                        tag = f" (median {core_med[core]:.0f}/patch)" if core in core_med.index else ""
                        ax.text(-0.20, 0.5, f"{RULE[rule]} {core}{tag}",
                                transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6, color=F.INK,
                                fontweight="bold")
                if rr == 0 and j == 0 and ci == len(cores) - 1:  # 100 um = 512 grid px, in the white corner of the first panel
                    x0 = lo_x + 0.02 * (hi_x - lo_x); y = hi_y - 0.03 * (hi_y - lo_y)
                    ax.plot([x0, x0 + GRID], [y, y], color=F.INK, lw=1.6, solid_capstyle="butt", clip_on=False)
                    ax.text(x0 + GRID / 2, y - 0.015 * (hi_y - lo_y), "100 um", color=F.INK, fontsize=5.5, ha="center", va="bottom", clip_on=False)
    if letter:
        F.panel_label(first, letter, x=-0.22, y=1.2)
    cor = cor.copy()
    cor["n_displayed"] = [counts.get((c, s)) for c, s in zip(cor.core, cor.set)]
    cor["median_per_eval_patch_displayed"] = [meds.get((c, s)) for c, s in zip(cor.core, cor.set)]
    return cor[cor.core.isin(cores)], pd.DataFrame(patch_counts)

