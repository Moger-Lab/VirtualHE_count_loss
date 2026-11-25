"""
Supplementary Figure S3 -- the frozen nucleus counter (used inside the
CG+M+C loss) and why a count, not an area.

(a) counter count vs StarDist count on the 496 validation-fold real patches
    (the counter was trained on training folds only).
(b) counter / StarDist per density bucket (median per-patch ratio).
(c) synthetic test (noise 0.02, empty-image offset removed): counter count
    and H-positive area for one nucleus, two touching nuclei and one nucleus
    of their combined area. Area cannot separate the last two; the count can.
(d) no counter exploitation: counter / StarDist count on the generated
    images, every evaluated checkpoint (held-out cores, epochs 27-30).
Sources: outputs\\s02_counter\\acceptance*.json / acceptance_val_per_patch.csv,
eval CSVs (n_fake_counter, n_fake). Model 3 parts of the old acceptance
test are not used.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

S02 = os.path.join(F.PROJECT, "outputs", "s02_counter")

if __name__ == "__main__":
    F.apply()
    V = pd.read_csv(os.path.join(S02, "acceptance_val_per_patch.csv"))
    A = json.load(open(os.path.join(S02, "acceptance.json")))
    D = json.load(open(os.path.join(S02, "acceptance_diagnostics.json")))
    fig, axs = plt.subplots(1, 4, figsize=(F.DOUBLE, 52 * F.MM), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 0.7, 1.1, 1]))

    ax = axs[0]
    ax.scatter(V.n_sd, V.counter, s=4, color=F.INK2, alpha=0.6, lw=0)
    m = max(V.n_sd.max(), V.counter.max()) * 1.05
    ax.plot([0, m], [0, m], color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_aspect("equal")
    ax.set_xlabel("StarDist nuclei"); ax.set_ylabel("counter estimate")
    ax.text(0.04, 0.96, f"r = {A['A']['pearson_r']:.3f}\nn = {len(V)} patches", transform=ax.transAxes, va="top",
            fontsize=6, color=F.INK2)
    F.panel_label(ax, "a", x=-0.38, y=1.04)

    ax = axs[1]
    tb = pd.DataFrame(A["A"]["table"])
    tb = tb[tb.bucket != "all"]
    ax.bar(range(3), tb.median_ratio, width=0.6, color="#c3c2b7", edgecolor="none")
    ax.axhline(1, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    for i, v in enumerate(tb.median_ratio):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=F.INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["sparse", "interm.", "dense"], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.8, 1.08); ax.set_ylabel("counter / StarDist"); ax.grid(axis="x", visible=False)
    F.panel_label(ax, "b", x=-0.55, y=1.04)

    ax = axs[2]
    syn = D["synthetic_noise_0.02"]; area = {k: v["hpos_area_fraction"] for k, v in A["C"]["configs"].items()}
    cfg = [("one", "one"), ("two_touching", "two\ntouching"), ("one_large_equal_area", "one,\n2x area")]
    w = 0.36
    cnt = [syn[k] for k, _ in cfg]
    ar = [area[k] / area["one"] for k, _ in cfg]
    ax.bar(np.arange(3) - w / 2, cnt, width=w, color=F.INK2, label="counter count")
    ax.bar(np.arange(3) + w / 2, ar, width=w, color="#c3c2b7", label="H-positive area (rel.)")
    for i, (c, a) in enumerate(zip(cnt, ar)):
        ax.text(i - w / 2, c + 0.04, f"{c:.2f}", ha="center", va="bottom", fontsize=5.5, color=F.INK2)
        ax.text(i + w / 2, a + 0.04, f"{a:.2f}", ha="center", va="bottom", fontsize=5.5, color=F.INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in cfg], fontsize=6)
    ax.set_ylim(0, 3.3); ax.set_ylabel("value"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.0, borderpad=0.2, ncol=1, labelspacing=0.2)
    F.panel_label(ax, "c", x=-0.38, y=1.04)

    ax = axs[3]
    rows = []
    for i, k in enumerate(F.KEYS):
        for t, s in zip(F.MODEL[k]["tags"], F.SEEDS):
            for e in F.EPS:
                d = pd.concat([pd.read_csv(os.path.join(F.RUNS_DIR, t, f"G_S2H_epoch{e:03d}_{sp}_eval.csv"))
                               for sp in ("val", "legacy_test")])
                rows.append(dict(model=k, run=t, seed=s, epoch=e, counter_over_stardist=d.n_fake_counter.mean() / d.n_fake.mean()))
    R = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    for i, k in enumerate(F.KEYS):
        v = R[R.model == k].counter_over_stardist.values
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=10, marker=F.MODEL[k]["marker"], color=F.MODEL[k]["color"],
                   edgecolor="white", lw=0.3)
    ax.axhline(1, color=F.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.axhspan(0.97, 1.03, color="#f0efec", lw=0, zorder=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(F.KEYS, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("counter / StarDist (fakes)"); ax.set_ylim(0.85, 1.15); ax.grid(axis="x", visible=False)
    ax.text(3.45, 1.035, "+/- 3%", fontsize=5.5, color=F.MUTED, ha="right", va="bottom")
    F.panel_label(ax, "d", x=-0.45, y=1.04)
    print(R.groupby("model").counter_over_stardist.agg(["min", "max"]))
    F.save(fig, "figS3_counter", {"a_val": V[["patch_uid", "core", "n_sd", "counter"]], "b_buckets": tb,
                                  "c_synthetic": pd.DataFrame(dict(config=[c[0] for c in cfg], counter=cnt, area_rel=ar)),
                                  "d_exploitation": R})
