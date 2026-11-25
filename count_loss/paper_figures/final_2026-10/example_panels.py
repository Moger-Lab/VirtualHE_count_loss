"""
Explanatory image panels for Figs 3 and 5 (data: g04_example_labels.py).

The four Fig 2 patches (sparse, intermediate, dense, stromal; seed 42, epoch
30) are shown again with the nucleus masks StarDist found, i.e. the objects
that are counted / measured to give the points of the quantitative panels.
All layout is in millimetres from the top-left of the figure, so blocks can be
stacked under existing GridSpec panels.

  outline_rows   Fig 3d  SRS input, adjacent H&E, four models; masks; count and fake/real
  size_block     Fig 5d  50 um crop, filled masks; per-patch median-area distribution below
  stain_block    Fig 5e  calibrated haematoxylin map; per-patch mean H distribution below
  empty_block    Fig 5f  12.5 um grid, cells with no real nucleus, generated nuclei in them
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
LAB = os.path.join(F.DATA, "example_labels")
UM2_PER_PX = (100.0 / 512) ** 2
OUTLINE = "#13b955"
FAB = "#d6191f"
MODEL_DIR = {"CG": "CG_full", "CG+M": "CGM_full", "CG+M+A": "A1_full", "CG+M+C": "A5_full"}
MODEL_TAB = dict(MODEL_DIR)


def has_images():
    return os.path.isdir(IMG) and os.path.isdir(LAB)


def read_img(d, uid):
    from PIL import Image
    return np.asarray(Image.open(os.path.join(IMG, d, f"{uid}.png")).convert("RGB"))


def read_lab(uid, s, key="labels"):
    with np.load(os.path.join(LAB, f"{uid}_{s}.npz")) as z:
        return z[key]


def polygons(lab):
    """Outline of every label as an (n, 2) array of (x, y) on the 512 px patch grid (pixel centres + 0.5)."""
    from skimage.measure import find_contours, regionprops
    out = []
    for p in regionprops(lab):
        r0, c0, r1, c1 = p.bbox
        m = np.pad(lab[r0:r1, c0:c1] == p.label, 1)
        cs = find_contours(m.astype(float), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        out.append(np.c_[c[:, 1] + c0 - 1 + 0.5, c[:, 0] + r0 - 1 + 0.5])
    return out


def draw_masks(ax, lab, color=OUTLINE, lw=0.45, fill=0.0, polys=None):
    P = polys if polys is not None else polygons(lab)
    if fill:
        ax.add_collection(PolyCollection(P, facecolors=color, alpha=fill, edgecolors="none", zorder=3))
    ax.add_collection(LineCollection(P, colors=color, linewidths=lw, zorder=4))
    return P


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def mm_axes(fig, H, x_mm, y_mm, w_mm, h_mm):
    """Axes with its top-left corner at (x_mm, y_mm) from the figure's top-left."""
    W = F.DOUBLE / F.MM
    return fig.add_axes([x_mm / W, 1 - (y_mm + h_mm) / H, w_mm / W, h_mm / H])


def model_marker(ax, k, y=1.17):
    ax.add_line(Line2D([0.5], [y], transform=ax.transAxes, marker=F.MODEL[k]["marker"], ms=4,
                       color=F.MODEL[k]["color"], markeredgecolor="white", markeredgewidth=0.4, clip_on=False))


def patch_rows(V):
    return {r.row: r for r in V.itertuples()}


# ------------------------------------------------------------------ Fig 3d
def outline_rows(fig, H, x0, y0, V, size=24.0, gap=3.0, rows=("sparse", "intermediate", "dense"), letter="d"):
    cols = [("srs", "SRS input", "srs", None), ("real", "adjacent real H&E", "real", "real")] + \
           [(k, k, MODEL_DIR[k], k) for k in F.KEYS]
    R = patch_rows(V)
    pitch = size + gap
    rec = []
    first = None
    for i, row in enumerate(rows):
        r = R[row]
        ry = y0 + 9 + i * (size + 8.5)
        n_real = int(r.n_real_orig)
        for j, (key, head, d, lk) in enumerate(cols):
            ax = mm_axes(fig, H, x0 + j * pitch, ry, size, size); blank(ax)
            first = first or ax
            ax.imshow(read_img(d, r.patch_uid), extent=(0, 512, 512, 0), interpolation="lanczos")
            ax.set_xlim(0, 512); ax.set_ylim(512, 0)
            if lk:
                lab = read_lab(r.patch_uid, "real" if lk == "real" else lk)
                draw_masks(ax, lab)
                n = int(lab.max())
                if lk == "real":
                    txt, fw = f"{n} nuclei", "bold"
                    col = F.INK
                else:
                    ratio = n / max(n_real, 1)
                    txt, fw = f"{n}  (x{ratio:.2f})", "normal"
                    col = F.INK2
                    rec.append(dict(row=row, patch_uid=r.patch_uid, model=lk, n_real=n_real, n_fake=n, ratio=ratio))
                ax.text(0.5, -0.03, txt, transform=ax.transAxes, ha="center", va="top", fontsize=6, color=col, fontweight=fw)
            if i == 0:
                ax.set_title(head, fontsize=6.5, color=F.INK, pad=4)
                if key in F.MODEL:
                    model_marker(ax, key, 1.2)
            if j == 0:
                ax.text(-0.07, 0.5, f"{row}\n({n_real} real)", transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=5.8, color=F.INK2, linespacing=1.15)
            if i == len(rows) - 1 and j == 0:  # 20 um scale bar in the SRS panel (512 px = 100 um)
                ax.plot([24, 24 + 102.4], [488, 488], color="white", lw=1.6, solid_capstyle="butt")
                ax.text(24 + 51.2, 476, "20 um", color="white", fontsize=5.5, ha="center", va="bottom")
            if i == len(rows) - 1 and j == 1:
                ax.add_line(Line2D([0.02, 0.18], [-0.30, -0.30], transform=ax.transAxes, color=OUTLINE, lw=1.0, clip_on=False))
                ax.text(0.22, -0.30, "StarDist nucleus mask: one mask = one counted nucleus", transform=ax.transAxes, fontsize=5.5,
                        color=F.INK2, va="center", ha="left")
    if letter:
        F.panel_label(first, letter, x=-0.42, y=1.3)
    return pd.DataFrame(rec)


# ------------------------------------------------------------------ Fig 5 helpers
CROP = 192  # px of the 512 px patch shown in Fig 5d (37.5 um)
FIG5_COLS =[("real", "real", "rt", "rt")] + [(k, k, MODEL_DIR[k], k) for k in F.KEYS]


def strip(fig, H, x0, y0, pitch, size, gap, h, series, ylabel, ylim, shown, fmt="{:.0f}"):
    """Violins aligned with the five image columns. series: {col_index: values}; shown: {col_index: value}."""
    ax = mm_axes(fig, H, x0 - gap / 2, y0, 5 * pitch, h)
    keys = ["real"] + F.KEYS
    for j, k in enumerate(keys):
        v = np.asarray(series[k], float); v = v[np.isfinite(v)]
        col = F.INK2 if k == "real" else F.MODEL[k]["color"]
        vp = ax.violinplot([v], positions=[j], widths=0.8, showextrema=False, showmedians=False)
        for b in vp["bodies"]:
            b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.35); b.set_linewidth(0.5)
        ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=col, lw=1.3, zorder=4)
        ax.scatter([j], [shown[k]], s=14, marker="o", facecolor="white", edgecolor=F.INK, linewidth=0.7, zorder=6)
    med_real = np.median(np.asarray(series["real"], float)[np.isfinite(series["real"])])
    ax.axhline(med_real, color=F.MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(*ylim); ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel, fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=5.5, length=1.8, pad=1.5)
    return ax


def _header(ax, head, k, size):
    ax.set_title(head, fontsize=6.2, color=F.INK, pad=3)
    if k in F.MODEL:
        model_marker(ax, k, 1 + 5.0 / size)


def crop_window(lab_real, size=256, stride=16):
    best, bw = -1, (0, 0)
    ys, xs = np.nonzero(lab_real)
    cy = np.array([np.mean(np.nonzero(lab_real == i)[0]) for i in range(1, int(lab_real.max()) + 1)])
    cx = np.array([np.mean(np.nonzero(lab_real == i)[1]) for i in range(1, int(lab_real.max()) + 1)])
    for y in range(0, 512 - size + 1, stride):
        for x in range(0, 512 - size + 1, stride):
            n = int(((cy >= y) & (cy < y + size) & (cx >= x) & (cx < x + size)).sum())
            if n > best:
                best, bw = n, (y, x)
    return bw


# ------------------------------------------------------------------ Fig 5d
def size_block(fig, H, x0, y0, V, pt_tab, size=24.0, gap=4.0, letter="d"):
    r = patch_rows(V)["dense"]
    uid = r.patch_uid
    pitch = size + gap
    chk = pd.read_csv(os.path.join(F.DATA, "fig_examples_checks.csv")).set_index(["patch_uid", "set"])
    wy, wx = crop_window(read_lab(uid, "rt"), CROP)
    rec, first = [], None
    ref = chk.loc[(uid, "rt"), "med_area_um2"]
    for j, (key, head, d, lk) in enumerate(FIG5_COLS):
        ax = mm_axes(fig, H, x0 + j * pitch, y0 + 5, size, size); blank(ax); first = first or ax
        ax.imshow(read_img(d, uid), extent=(0, 512, 512, 0), interpolation="lanczos")
        lab = read_lab(uid, "rt" if lk == "rt" else lk)
        draw_masks(ax, lab, fill=0.28, lw=0.5)
        ax.set_xlim(wx, wx + CROP); ax.set_ylim(wy + CROP, wy)
        _header(ax, head, key, size)
        a = chk.loc[(uid, "rt" if lk == "rt" else lk), "med_area_um2"]
        txt = f"{a:.1f} um$^2$" if lk == "rt" else f"{a:.1f} um$^2$\n(x{a / ref:.2f})"
        ax.text(0.5, -0.03, txt, transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
                color=F.INK if lk == "rt" else F.INK2, fontweight="bold" if lk == "rt" else "normal")
        rec.append(dict(patch_uid=uid, model=key, patch_median_area_um2=a, ratio_to_real=a / ref))
        if j == 0:
            ax.text(-0.08, 0.5, f"dense patch\n37 um crop", transform=ax.transAxes, rotation=90, ha="right", va="center",
                    fontsize=5.8, color=F.INK2, linespacing=1.15)
            ax.plot([wx + 12, wx + 12 + 51.2], [wy + CROP - 12] * 2, color=F.INK, lw=1.6, solid_capstyle="butt")
            ax.text(wx + 12 + 25.6, wy + CROP - 17, "10 um", color=F.INK, fontsize=5.2, ha="center", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    T = pt_tab
    m = T.bucket == "high"
    series = {"real": T.rt_area[m]}
    series.update({k: T[f"{k}_area"][m] for k in F.KEYS})
    shown = {"real": ref}
    shown.update({k: chk.loc[(uid, k), "med_area_um2"] for k in F.KEYS})
    ax = strip(fig, H, x0, y0 + 5 + size + 10.5, pitch, size, gap, 10.5, series, "median area\nper patch (um$^2$)", (8, 40), shown)
    ax.text(0.99, 0.97, f"all {int(m.sum())} dense patches; o = patch above", transform=ax.transAxes, fontsize=5.3,
            color=F.MUTED, ha="right", va="top")
    if letter:
        F.panel_label(first, letter, x=-9.0 / size, y=1 + 7.5 / size)
    return pd.DataFrame(rec)


# ------------------------------------------------------------------ Fig 5e
def stain_block(fig, H, x0, y0, V, pt_tab, size=24.0, gap=4.0, letter="e"):
    from matplotlib import cm
    r = patch_rows(V)["stromal"]
    uid = r.patch_uid
    pitch = size + gap
    chk = pd.read_csv(os.path.join(F.DATA, "fig_examples_checks.csv")).set_index(["patch_uid", "set"])
    ref = chk.loc[(uid, "rt"), "meanH_tissue"]
    rec, first = [], None
    for j, (key, head, d, lk) in enumerate(FIG5_COLS):
        s = "rt" if lk == "rt" else lk
        ax = mm_axes(fig, H, x0 + j * pitch, y0 + 5, size, size); blank(ax); first = first or ax
        Hm = read_lab(uid, s, "H").astype(np.float32)
        ax.imshow(np.clip(Hm / 0.10, 0, 1), cmap=cm.Purples, vmin=0, vmax=1, extent=(0, 512, 512, 0), interpolation="bilinear")
        from skimage.measure import find_contours
        t = read_lab(uid, s, "tissue")
        for c in find_contours(np.pad(t, 1).astype(float), 0.5):
            ax.plot(c[:, 1] - 1 + 0.5, c[:, 0] - 1 + 0.5, color=F.INK2, lw=0.35)
        ax.set_xlim(0, 512); ax.set_ylim(512, 0)
        _header(ax, head, key, size)
        h = chk.loc[(uid, s), "meanH_tissue"]
        txt = f"H {h:.3f}" if lk == "rt" else f"H {h:.3f}\n(x{h / ref:.2f})"
        ax.text(0.5, -0.03, txt, transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
                color=F.INK if lk == "rt" else F.INK2, fontweight="bold" if lk == "rt" else "normal")
        rec.append(dict(patch_uid=uid, model=key, mean_H_tissue=h, ratio_to_real=h / ref))
        if j == 0:
            ax.text(-0.08, 0.5, "stromal patch\nH channel", transform=ax.transAxes, rotation=90, ha="right", va="center",
                    fontsize=5.8, color=F.INK2, linespacing=1.15)
    T = pt_tab
    m = T.stromal == 1
    series = {"real": T.rt_H[m]}
    series.update({k: T[f"{k}_H"][m] for k in F.KEYS})
    shown = {"real": ref}
    shown.update({k: chk.loc[(uid, k), "meanH_tissue"] for k in F.KEYS})
    ax = strip(fig, H, x0, y0 + 5 + size + 10.5, pitch, size, gap, 10.5, series, "mean H in tissue\nper patch", (0.0, 0.09), shown)
    ax.text(0.99, 0.97, f"all {int(m.sum())} stromal patches; o = patch above", transform=ax.transAxes, fontsize=5.3,
            color=F.MUTED, ha="right", va="top")
    if letter:
        F.panel_label(first, letter, x=-9.0 / size, y=1 + 7.5 / size)
    return pd.DataFrame(rec)


# ------------------------------------------------------------------ Fig 5f
def centroids(lab):
    from skimage.measure import regionprops
    P = regionprops(lab)
    return np.array([p.centroid for p in P]).reshape(-1, 2)


def empty_block(fig, H, x0, y0, V, size=24.0, gap=4.0, letter="f"):
    from matplotlib.patches import Rectangle
    r = patch_rows(V)["stromal"]
    uid = r.patch_uid
    pitch = size + gap
    chk = pd.read_csv(os.path.join(F.DATA, "fig_examples_checks.csv")).set_index(["patch_uid", "set"])
    lab_real = read_lab(uid, "real")
    cr = centroids(lab_real)
    cell = 64
    cnt = np.zeros((8, 8), int)
    for y, x in cr:
        cnt[min(int(y // cell), 7), min(int(x // cell), 7)] += 1
    empty = cnt == 0
    n_empty = int(empty.sum())
    assert n_empty == int(chk.loc[(uid, "CG"), "n_empty_cells"]), "empty-cell count differs from the evaluation"
    n_real = int(lab_real.max())
    rec, first = [], None
    cols = [("real", "real H&E", "real", "real")] + [(k, k, MODEL_DIR[k], k) for k in F.KEYS]
    for j, (key, head, d, lk) in enumerate(cols):
        ax = mm_axes(fig, H, x0 + j * pitch, y0 + 5, size, size); blank(ax); first = first or ax
        ax.imshow(read_img(d, uid), extent=(0, 512, 512, 0), interpolation="lanczos")
        for a, b in zip(*np.nonzero(empty)):
            ax.add_patch(Rectangle((b * cell, a * cell), cell, cell, fc="#ffd24a", ec="none", alpha=0.30, zorder=2))
        for g in range(1, 8):
            ax.plot([g * cell] * 2, [0, 512], color="white", lw=0.3, alpha=0.8, zorder=2)
            ax.plot([0, 512], [g * cell] * 2, color="white", lw=0.3, alpha=0.8, zorder=2)
        lab = read_lab(uid, lk)
        P = polygons(lab)
        cents = centroids(lab)
        inside = np.array([empty[min(int(y // cell), 7), min(int(x // cell), 7)] for y, x in cents]) if len(cents) else np.array([], bool)
        # mask outlines: green where the nucleus is in a cell with real nuclei, red where it is in a cell empty in the real section
        from skimage.measure import regionprops
        ids = [p.label for p in regionprops(lab)]
        col = [FAB if (lk != "real" and inside[i]) else OUTLINE for i in range(len(ids))]
        ax.add_collection(LineCollection(P, colors=col, linewidths=0.55, zorder=4))
        ax.set_xlim(0, 512); ax.set_ylim(512, 0)
        _header(ax, head, key, size)
        if lk == "real":
            txt, color_, fw = f"{n_empty} of 64 cells empty", F.INK, "bold"
        else:
            nin = int(inside.sum())
            txt, color_, fw = f"{nin} in empty cells ({nin / n_empty:.2f}/cell)", F.INK2, "normal"
            rec.append(dict(patch_uid=uid, model=key, n_fake=len(ids), n_in_empty_cells=nin, n_empty_cells=n_empty,
                            fabrication_per_empty_cell=nin / n_empty, output_level=len(ids) / n_real))
        ax.text(0.5, -0.03, txt, transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color=color_, fontweight=fw)
        if j == 0:
            ax.text(-0.08, 0.5, "stromal patch\n12.5 um cells", transform=ax.transAxes, rotation=90, ha="right", va="center",
                    fontsize=5.8, color=F.INK2, linespacing=1.15)
        if j == 4:
            ax.add_patch(Rectangle((1.04, 0.58), 0.09, 0.09, transform=ax.transAxes, fc="#ffd24a", alpha=0.5, ec="none", clip_on=False))
            ax.text(1.17, 0.625, "cell with no\nreal nucleus", transform=ax.transAxes, fontsize=5.3, color=F.INK2, va="center", ha="left")
            ax.add_line(Line2D([1.045, 1.135], [0.40, 0.40], transform=ax.transAxes, color=FAB, lw=1.0, clip_on=False))
            ax.text(1.17, 0.40, "generated\nnucleus in\nsuch a cell", transform=ax.transAxes, fontsize=5.3, color=F.INK2, va="center", ha="left")
    if letter:
        F.panel_label(first, letter, x=-9.0 / size, y=1 + 7.5 / size)
    return pd.DataFrame(rec)


def patch_table():
    """Per-patch seed-42 epoch-30 values for the strips (stageA tables), keyed like patches.csv."""
    T = pd.read_csv(os.path.join(F.STAGEA, "patches.csv")).set_index("patch_uid")
    tab = os.path.join(F.STAGEA, "tables", "stardist")
    for name, f in [("rt", "rt")] + [(k, MODEL_TAB[k]) for k in F.KEYS]:
        t = pd.read_csv(os.path.join(tab, f"{f}_patches.csv")).set_index("patch_uid")
        T[f"{name}_area"] = t.med_area_um2.reindex(T.index)
        T[f"{name}_H"] = t.mean_H_tissue.reindex(T.index)
    return T



