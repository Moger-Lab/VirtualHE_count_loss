"""
fcommon -- shared style, names and data access for the final paper figures
(handoff 2026-10-02, DECISION_LOG D-33 onwards). Independent of
paper_figures\\style.py and ladder.py, which keep their old labels.

Models: CG, CG+M, CG+M+A, CG+M+C (three seeds each, 42 / 7 / 123).
Evaluation: StarDist, per-patch fake count averaged over epochs 27-30, real
baseline = real after the rgb_to_he / he_to_rgb round trip ("rt"), density
buckets fixed by the real StarDist count on the original RGB (as s18).
Held-out evaluation cores: 17 cores, 1038 patches.

Colours (validated with palette_check.py, all pairs, white surface; D-34):
CG #4a3aa7 violet circle, CG+M #eda100 yellow square, CG+M+A #a09e97 grey
diamond, CG+M+C #2a78d6 blue triangle. The grey is a deliberate neutral and
the yellow is below 3:1 contrast: identity is always also carried by the
marker shape and a direct label or legend.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(_here))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s18i_stageA_report import m6, slope  # identical metric definitions to s18

OUT = os.environ.get("VHE_FIG_OUT", _here)  # repository: write regenerated outputs elsewhere
DATA = os.path.join(OUT, "data")  # repository: follows VHE_FIG_OUT (default unchanged)
os.makedirs(DATA, exist_ok=True)
STAGEA = __import__("vhe_paths").work("s18_metric_screen", "stageA")
RUNS_DIR = __import__("vhe_paths").work("runs")
EPS = [27, 28, 29, 30]
N_BOOT = 1000

MM = 1 / 25.4
SINGLE = 85 * MM
DOUBLE = 174 * MM
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"

KEYS = ["CG", "CG+M", "CG+M+A", "CG+M+C"]
MODEL = {
    "CG": dict(color="#4a3aa7", marker="o", tags=["CG_full", "CG_full_s7", "CG_full_s123"]),
    "CG+M": dict(color="#eda100", marker="s", tags=["CGM_full", "CGM_full_s7", "CGM_full_s123"]),
    "CG+M+A": dict(color="#a09e97", marker="D", tags=["A1_full", "A1_full_s7", "A1_full_s123"]),
    "CG+M+C": dict(color="#2a78d6", marker="^", tags=["A5_full", "A5_full_s7", "A5_full_s123"]),
}
SEEDS = [42, 7, 123]
BUCKET_LABEL = {"low": "sparse (< 33)", "med": "intermediate (33-55)", "high": "dense (>= 56)"}


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titleweight": "normal",
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel_label(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, name, sources=None, sub=""):
    """PDF + SVG (vector) + PNG preview + one source CSV per panel. Refuses to
    overwrite unless FIG_REDRAFT=1 (redrafting this folder's own outputs; the
    target is always inside final_2026-10)."""
    d = os.path.join(OUT, sub) if sub else OUT
    assert os.path.abspath(d).startswith(os.path.abspath(OUT))
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        p = os.path.join(d, f"{name}.{ext}")
        if os.path.exists(p) and os.environ.get("FIG_REDRAFT") != "1":
            raise FileExistsError(p)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight")
    for key, df in (sources or {}).items():
        df.to_csv(os.path.join(d, f"{name}_{key}_source.csv"), index=False)
    plt.close(fig)
    print("wrote", os.path.join(d, name))


# ------------------------------------------------------------------ data
def patches():
    """patch_uid, core, split, bucket (original-RGB StarDist), stromal, n_real (orig), n_real_rt."""
    pt = pd.read_csv(os.path.join(STAGEA, "patches.csv"))
    rt = pd.read_csv(os.path.join(STAGEA, "tables", "stardist", "rt_patches.csv"))
    assert list(rt.patch_uid) == list(pt.patch_uid)
    pt["n_real_rt"] = rt.n.values.astype(float)
    return pt


def run_table(tag, uids):
    """Per-patch fake quantities for one run, averaged over epochs 27-30, in uids order."""
    per = []
    for e in EPS:
        d = pd.concat([pd.read_csv(os.path.join(RUNS_DIR, tag, f"G_S2H_epoch{e:03d}_{s}_eval.csv"))
                       for s in ("val", "legacy_test")]).set_index("patch_uid").loc[uids]
        per.append(d)
    out = pd.DataFrame({"patch_uid": uids})
    for c in ("n_fake", "fake_in_empty"):
        out[c] = np.mean([d[c].values.astype(float) for d in per], axis=0)
        for e, d in zip(EPS, per):
            out[f"{c}_e{e}"] = d[c].values.astype(float)
    out["n_empty_cells"] = per[0]["n_empty_cells"].values.astype(float)
    return out


def load_all():
    pt = patches()
    uids = pt.patch_uid.values
    R = {t: run_table(t, uids) for k in KEYS for t in MODEL[k]["tags"]}
    return pt, R


def run_metrics(nf, nr, bucket, fie=None, nec=None):
    """r, slope, disc (s18 m6), median fake/real, dense-bucket ratio, stromal fabrication."""
    o = m6(nf, nr, bucket)
    ratio = nf / np.maximum(nr, 1)
    o["fake_real"] = float(np.median(ratio))
    o["ratio_high"] = float(np.median(ratio[bucket == "high"]))
    if fie is not None:
        o["stromal_fab"] = float(fie.sum() / max(nec.sum(), 1))
    return o


def boot_indices(pt, n=N_BOOT, seed=0):
    """Core-cluster bootstrap indices, built exactly as s18k (sorted cores, rng seed 0)."""
    cores = pt.core.values
    core_idx = {c: np.nonzero(cores == c)[0] for c in np.unique(cores)}
    ucores = np.array(sorted(core_idx))
    rng = np.random.default_rng(seed)
    return [np.concatenate([core_idx[c] for c in rng.choice(ucores, len(ucores))]) for _ in range(n)]


def seed_mean_fake(R, key):
    return np.mean([R[t].n_fake.values for t in MODEL[key]["tags"]], axis=0)


def legend_handles(keys=KEYS, ms=4.5):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=MODEL[k]["color"], marker=MODEL[k]["marker"], ls="", ms=ms,
                   markeredgecolor="white", markeredgewidth=0.4, label=k) for k in keys]
