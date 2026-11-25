"""
Supplementary Figure S1 -- registration of the adjacent H&E section onto the
SRS grid (v2 pipeline; raw data and v2 outputs read-only).

(a) pipeline strip (built once): core segmentation (circle fit) -> cohort
    scale 1.02317 -> left-right mirror -> circle-centre alignment -> local
    rigid refinement -> residual check against a mismatched-core null.
(b) before / after examples, one core per density class (core median real
    nuclei per 100 um patch, tertiles over registration-tier-A cores):
    SRS (channel 1) | H&E as scanned | H&E after registration |
    checkerboard of SRS and registered H&E (inverted luminance, ~100 um
    squares) | 2x detail of the checkerboard centre.
    Candidate versions A-D: rule, not model output: tier A, residual <=
    cohort median, scan coverage >= 0.9; ranked by SRS-H&E architecture r at
    50 um; version X takes rank X in each class. E: unselected (seed 0).
Outputs: figS1a_pipeline, candidates\\figS1b_candidate_*, contact sheet.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F
import countlib as cl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RAW_HE = os.path.join(__import__("vhe_paths").RAW_DATA, "H&E")
DS = 4
SQ = 110  # checkerboard square, px at DS (~100 um)
ROWS = ["sparse", "intermediate", "dense"]


def raw_he(core):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    for n in (f"{core}_H&E.png", f"{core}.png"):
        p = os.path.join(RAW_HE, n)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            im.thumbnail((1200, 1200), Image.LANCZOS)
            return np.asarray(im)
    raise FileNotFoundError(core)


def aligned(core):
    srs, he = cl.core_arrays(core)
    valid = np.asarray(np.load(os.path.join(cl.ALIGNED_DIR, f"{core}_valid.npy"), mmap_mode="r")[::DS, ::DS])
    s = np.asarray(srs[1, ::DS, ::DS]).astype(np.float32)
    lo, hi = np.percentile(s[valid], [0.5, 99.5])
    s = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)
    h = np.asarray(he[::DS, ::DS])
    return s, h, valid


def checker(s, h, valid):
    lum = 1 - h.astype(np.float32).mean(-1) / 255.0
    lo, hi = np.percentile(lum[valid], [0.5, 99.5])
    lum = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)
    yy, xx = np.indices(s.shape)
    m = ((yy // SQ + xx // SQ) % 2).astype(bool)
    out = np.where(m, s, lum)
    out[~valid] = 0
    return out


def strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def pipeline():
    steps = ["segment core\n(circle fit, both\nmodalities)", "scale H&E\n(cohort 1.02317\nH&E px / SRS px)",
             "mirror H&E\nleft-right", "align circle\ncentres", "local rigid\nrefinement\n(dy, dx)",
             "residual and\nmatched vs\nmismatched null"]
    fig, ax = plt.subplots(figsize=(F.DOUBLE, 26 * F.MM)); ax.axis("off")
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1)
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.12), 0.8, 0.76, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc="#f0efec", ec=F.AXIS, lw=0.5))
        ax.text(i + 0.46, 0.5, s, ha="center", va="center", fontsize=6, color=F.INK, linespacing=1.15)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.06, 0.5), xytext=(i + 0.86, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=F.INK2, lw=0.7, mutation_scale=6))
    F.panel_label(ax, "a", x=-0.01, y=0.95)
    F.save(fig, "figS1a_pipeline")


def build(V, title):
    fig, axs = plt.subplots(len(ROWS), 5, figsize=(F.DOUBLE, 112 * F.MM), gridspec_kw=dict(wspace=0.04, hspace=0.18))
    heads = ["SRS (input channel)", "H&E as scanned", "H&E registered", "checkerboard\nSRS / H&E", "detail (2x)"]
    for i, row in enumerate(ROWS):
        r = V[V.row == row].iloc[0]
        s, h, valid = aligned(r.core)
        ys, xs = np.nonzero(valid); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cb = checker(s, h, valid)
        hh = h.copy(); hh[~valid] = 255
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; d = (y1 - y0) // 4
        ims = [(s[y0:y1, x0:x1], "gray"), (raw_he(r.core), None), (hh[y0:y1, x0:x1], None),
               (cb[y0:y1, x0:x1], "gray"), (cb[cy - d:cy + d, cx - d:cx + d], "gray")]
        for j, (im, cm) in enumerate(ims):
            ax = axs[i, j]
            ax.imshow(im, cmap=cm, interpolation="lanczos")
            strip_axes(ax)
            if i == 0:
                ax.set_title(heads[j], fontsize=6.3, color=F.INK, pad=3)
        axs[i, 0].text(-0.06, 0.5, f"{row}\n{r.core}\n(median {r.median_nuclei:.0f} nuclei,\nresidual {r.resid_um:.1f} um)",
                       transform=axs[i, 0].transAxes, rotation=90, ha="right", va="center", fontsize=5.6, color=F.INK2,
                       linespacing=1.1)
    if title:
        fig.suptitle(title, x=0.12, ha="left", fontsize=7, color=F.INK2, y=0.995)
    F.panel_label(axs[0, 0], "b", x=-0.32, y=1.12)
    return fig


if __name__ == "__main__":
    F.apply()
    pipeline()
    T = pd.read_csv(os.path.join(cl.V2_RESULTS, "16_transforms.csv"))
    D = pd.read_csv(os.path.join(cl.V2_RESULTS, "09_nuclear_density.csv"))[["core", "median"]].rename(columns={"median": "median_nuclei"})
    T = T.merge(D, on="core")
    A = T[T.tier == "A_registered"].copy()
    q = A.median_nuclei.quantile([1 / 3, 2 / 3]).values
    A["row"] = np.where(A.median_nuclei < q[0], "sparse", np.where(A.median_nuclei < q[1], "intermediate", "dense"))
    el = A[(A.resid_um <= T.resid_um.median()) & (A.scan_coverage >= 0.9)]
    vers = []
    for row in ROWS:
        c = el[el.row == row].sort_values("r_50um", ascending=False)
        for k, v in enumerate("ABCD"):
            vers.append(dict(version=v, row=row, **c.iloc[k][["core", "median_nuclei", "resid_um", "r_50um"]].to_dict()))
        rng = np.random.default_rng(0)
        pool = A[A.row == row]
        vers.append(dict(version="E (unselected)", row=row,
                         **pool.iloc[rng.integers(len(pool))][["core", "median_nuclei", "resid_um", "r_50um"]].to_dict()))
    C = pd.DataFrame(vers)
    C.to_csv(os.path.join(F.DATA, "figS1_candidates.csv"), index=False)
    print(C.to_string(index=False))
    pngs = []
    for v in ["A", "B", "C", "D", "E (unselected)"]:
        fig = build(C[C.version == v], f"Figure S1b candidate {v}")
        F.save(fig, f"figS1b_candidate_{v[0]}", {"cores": C[C.version == v]}, sub="candidates")
        pngs.append((v, os.path.join(F.OUT, "candidates", f"figS1b_candidate_{v[0]}.png")))
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for _, p in pngs]
    ims = [i.resize((i.width // 2, i.height // 2), Image.LANCZOS) for i in ims]
    W = sum(i.width for i in ims) + 40 * (len(ims) + 1); H = max(i.height for i in ims) + 90
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arialbd.ttf", 44)
    x = 40
    for (v, _), im in zip(pngs, ims):
        d.text((x, 20), v, fill=(11, 11, 11), font=font); sheet.paste(im, (x, 80)); x += im.width + 40
    sheet.save(os.path.join(F.OUT, "candidates", "figS1b_contact_sheet.png"))
    print("wrote contact sheet")
