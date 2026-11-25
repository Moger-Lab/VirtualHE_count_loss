"""
s02b -- per-core SRS normalisation cache (flag SRS_NORM_MODE='per_core').

p1 / p99.5 per channel over tissue pixels of the full cube, once per core.
Tissue = SRS-only mask (CARS above a corner-estimated background), so it is
independent of registration and valid for unpaired cores. Sampled at stride
4 (1.2M pixels per core), which pins percentiles far below the level of
patch-to-patch variation.

Output: D:\vhe_count_work\srs_norm\per_core_percentiles.csv (+ a copy in
outputs/s02_counter/ for the record) and a mask sanity figure.
"""

import os
import shutil
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
    cores = sorted(tr["core"].unique())
    print(f"Computing per-core SRS percentiles for {len(cores)} cores...")
    df = cl.compute_per_core_percentiles(cores)
    out_dir = os.path.join(_script_dir, "outputs", "s02_counter")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(cl.SRS_NORM_CACHE, os.path.join(out_dir, "per_core_percentiles.csv"))
    # Cohort-global range per channel, pooled over TRAINING cores only
    # (rotating, fold != 0): used for the tissue-type channels (SHG, TPEF)
    # under TYPE_CHANNEL_NORM='global' (DECISION_LOG D-06).
    train_cores = sorted(cv[(cv.group == "rotating") & (cv.fold != 0)].core)
    pooled = {c: [] for c in range(4)}
    for core in train_cores:
        srs, _ = cl.core_arrays(core)
        mask = cl.srs_tissue_mask(srs, 8)
        for c in range(4):
            pooled[c].append(np.asarray(srs[c, ::8, ::8])[mask].astype(np.float32))
    g = dict(core="__GLOBAL__", tissue_frac=np.nan)
    for c in range(4):
        lo, hi = np.percentile(np.concatenate(pooled[c]), cl.PER_CORE_PERCENTILES)
        g[f"ch{c}_lo"] = float(lo); g[f"ch{c}_hi"] = float(hi)
    print(f"GLOBAL (pooled over {len(train_cores)} training cores): "
          + "  ".join(f"ch{c} [{g[f'ch{c}_lo']:.0f}, {g[f'ch{c}_hi']:.0f}]" for c in range(4)))
    import pandas as pd
    df = pd.concat([df, pd.DataFrame([g])], ignore_index=True)
    df.to_csv(cl.SRS_NORM_CACHE, index=False)
    shutil.copy(cl.SRS_NORM_CACHE, os.path.join(out_dir, "per_core_percentiles.csv"))
    df = df[df.core != "__GLOBAL__"]
    for c in range(4):
        hi = df[f"ch{c}_hi"]
        print(f"ch{c}: p99.5 across cores median {hi.median():.0f}, range {hi.min():.0f}-{hi.max():.0f} "
              f"(CV {hi.std() / hi.mean():.2f})")
    print(f"tissue fraction median {df.tissue_frac.median():.2f}, min {df.tissue_frac.min():.2f} ({df.loc[df.tissue_frac.idxmin(), 'core']})")

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for j, core in enumerate(["D-3", "B-5", "G-1", "B-9"]):
        srs, _ = cl.core_arrays(core)
        mask = cl.srs_tissue_mask(srs)
        ax[0, j].imshow(np.asarray(srs[2, ::4, ::4]), cmap="gray", vmax=np.percentile(srs[2, ::8, ::8], 99.5))
        ax[0, j].set_title(f"{core} CARS"); ax[1, j].imshow(mask); ax[1, j].set_title(f"tissue mask {mask.mean():.2f}")
        for a in ax[:, j]:
            a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "srs_tissue_masks.png"), dpi=80); plt.close(fig)
