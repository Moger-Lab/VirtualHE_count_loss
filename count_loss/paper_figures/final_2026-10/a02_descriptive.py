"""
a02 -- descriptive by-density-class values (M1 nuclear area, M5 stain) with
per-seed values and core-cluster bootstrap CIs, for Fig 5a/b and S7; and the
per-checkpoint level vs stromal fabrication table for Fig 5c (analysis 3).

M1/M5: epoch 30 (s18 stage A tables), real round-trip baseline, per-patch
fake/real ratio (s18i ratio_getter), median over the patches of a class,
mean over seeds. Classes: sparse / intermediate / dense (original-RGB real
StarDist count) and stromal (s18 rule). Point values reproduce
outputs\\s18_metric_screen\\stageA\\by_density_class.csv (checked below).
These are descriptive only (they fail the seed-noise rule, D-30) and are
never used to rank models.

Level vs fabrication: every evaluated checkpoint (4 models x 3 seeds x
epochs 27-30 = 48); level = median per-patch fake/real (rt), fabrication =
fake nuclei per real-empty 12.5 um cell (eval CSVs).
Outputs (data\\): descriptive_by_class.csv, descriptive_by_class_per_run.csv,
checkpoint_level_fabrication.csv. Never overwrites.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd

MEAS = [  # metric label, detector, column
    ("M1 nuclear area", "stardist", "med_area_um2"),
    ("M1 nuclear area", "cellvit_sam", "med_area_um2"),
    ("M5 mean H (tissue)", "stardist", "mean_H_tissue"),
    ("M5 mean E (tissue)", "stardist", "mean_E_tissue"),
    ("M5 H fraction", "stardist", "H_frac"),
]
CLASSES = ["low", "med", "high", "stromal"]


def ratio(det, col, tag):
    f = pd.read_csv(os.path.join(F.STAGEA, "tables", det, f"{tag}_patches.csv"))[col].values.astype(float)
    r = pd.read_csv(os.path.join(F.STAGEA, "tables", det, "rt_patches.csv"))[col].values.astype(float)
    return np.where((r > 0) & np.isfinite(r) & np.isfinite(f), f / np.where(r > 0, r, 1), np.nan)


if __name__ == "__main__":
    outs = [os.path.join(F.DATA, n) for n in ("descriptive_by_class.csv", "descriptive_by_class_per_run.csv",
                                              "checkpoint_level_fabrication.csv")]
    if any(os.path.exists(p) for p in outs):
        sys.exit("a02 outputs exist; refusing to overwrite")
    pt = F.patches()
    masks = {c: (pt.bucket.values == c) for c in CLASSES[:3]}
    masks["stromal"] = pt.stromal.values.astype(bool)
    bidx = F.boot_indices(pt)

    rows, per = [], []
    for label, det, col in MEAS:
        X = {t: ratio(det, col, t) for k in F.KEYS for t in F.MODEL[k]["tags"]}
        for cl in CLASSES:
            m = masks[cl]
            if label.startswith("M1") and cl == "stromal":
                continue
            for k in F.KEYS:
                tags = F.MODEL[k]["tags"]
                v = [float(np.nanmedian(X[t][m])) for t in tags]
                for t, s, x in zip(tags, F.SEEDS, v):
                    per.append(dict(metric=label, detector=det, subset=cl, model=k, run=t, seed=s, value=x))
                bs = [np.mean([np.nanmedian(X[t][idx][m[idx]]) for t in tags]) for idx in bidx]
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append(dict(metric=label, detector=det, subset=cl, model=k, value=float(np.mean(v)),
                                 ci_lo=lo, ci_hi=hi, n_patches=int(m.sum())))
    T = pd.DataFrame(rows)

    # check against s18 by_density_class.csv (rt baseline)
    ref = pd.read_csv(os.path.join(F.PROJECT, "outputs", "s18_metric_screen", "stageA", "by_density_class.csv"))
    ref = ref[ref.baseline == "rt"]
    ref["detector"] = ref.detector.replace({"none": "stardist"})
    mm = T.merge(ref, on=["metric", "detector", "subset", "model"], suffixes=("", "_s18"))
    print(f"check vs by_density_class.csv: {len(mm)} rows matched, max |diff| {np.abs(mm.value - mm.value_s18).max():.2e}")

    T.to_csv(outs[0], index=False)
    pd.DataFrame(per).to_csv(outs[1], index=False)

    # per-checkpoint level vs stromal fabrication
    pt2, R = F.load_all()
    nr = pt2.n_real_rt.values
    ck = []
    for k in F.KEYS:
        for t, s in zip(F.MODEL[k]["tags"], F.SEEDS):
            r = R[t]
            for e in F.EPS:
                nf = r[f"n_fake_e{e}"].values
                ck.append(dict(model=k, run=t, seed=s, epoch=e, level=float(np.median(nf / np.maximum(nr, 1))),
                               stromal_fab=float(r[f"fake_in_empty_e{e}"].sum() / max(r.n_empty_cells.sum(), 1))))
    C = pd.DataFrame(ck)
    C.to_csv(outs[2], index=False)
    print("checkpoints per model:", C.groupby("model").size().to_dict())
    print("within-model r(level, fabrication):",
          {k: round(float(np.corrcoef(g.level, g.stromal_fab)[0, 1]), 3) for k, g in C.groupby("model")})
    print("pooled r:", round(float(np.corrcoef(C.level, C.stromal_fab)[0, 1]), 3))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(T.to_string(index=False))
