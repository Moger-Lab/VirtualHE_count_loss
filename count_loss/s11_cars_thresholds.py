"""
s11 -- per-core CARS tissue thresholds for the SRS-emptiness loss (A6,
D-18), plus a visual check of the SRS-empty mask on legacy C-3 / B-8 tiles
and a count of how much of a typical training patch is SRS-empty.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import countlib as cl

if __name__ == "__main__":
    m, cv, tr = cl.load_v2_tables()
    cl.compute_cars_thresholds(sorted(tr.core.unique()))
    m["split"] = cl.split_of(m, 0)
    rng = np.random.default_rng(0)
    trn = m[(m.split == "train") & (m.role == "train")]
    fr = []
    for r in trn.iloc[rng.choice(len(trn), 300, replace=False)].itertuples():
        s12, _ = cl.v2_patch_native(r.core, r.y0, r.x0, 441, channels=[1, 2])
        fr.append(cl.srs_empty_mask(r.core, s12).mean())
    fr = np.array(fr)
    print(f"SRS-empty fraction over 300 training patches: mean {fr.mean():.3f}, median {np.median(fr):.3f}, "
          f"patches with >5% empty {np.mean(fr > 0.05):.2f}")
    leg = m[(m.split == "legacy_test") & (m.role == "eval")]
    picks = list(leg[leg.core == "C-3"].iloc[[3, 12, 25, 40]].itertuples()) + list(leg[leg.core == "B-8"].iloc[[5, 20]].itertuples()) + list(leg[leg.core == "F-6"].iloc[[4, 15]].itertuples())
    fig, ax = plt.subplots(3, len(picks), figsize=(3 * len(picks), 9))
    for j, r in enumerate(picks):
        s12, he = cl.v2_patch_native(r.core, r.y0, r.x0, 441, channels=[1, 2])
        cars = s12[1:2]
        em = cl.srs_empty_mask(r.core, s12)
        ax[0, j].imshow(cars[0], cmap="gray", vmax=np.percentile(cars[0], 99.5)); ax[0, j].set_title(f"{r.core} CARS")
        ax[1, j].imshow(cl.resize_rgb(he)); ax[1, j].set_title("real H&E (serial section)")
        ov = cl.resize_rgb(he).astype(float) / 255; ov[em > 0] = ov[em > 0] * 0.4 + np.array([0, 0.6, 0.9]) * 0.6
        ax[2, j].imshow(ov); ax[2, j].set_title(f"SRS-empty (blue) {em.mean():.2f}")
        for a in ax[:, j]:
            a.axis("off")
    plt.tight_layout()
    p = os.path.join(_script_dir, "outputs", "s09_gallery", "srs_empty_mask_check.png")
    plt.savefig(p, dpi=90); print(p)
