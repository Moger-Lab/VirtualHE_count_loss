"""
s02a -- StarDist cache on real v2 H&E patches (Step 2, part 1).

01_data_pipeline.py's Stage 1c only derives a pixel threshold from 30
patches; it stores no per-patch detections, so it cannot be reused here.
This runs v1's exact protocol (2D_versatile_he, percentiles 1.0/99.8, min
area 10 px, 100 um at 512 px) once per patch and caches centroids and areas
already on the 512 network grid.

Rows cached (screening design, val fold 0):
  split train, role train     (counter + generator training)
  split val,   role eval      (disjoint tiling, used for all validation)
  split legacy_test, role eval (kept for the final Model 3 comparison only)

Sharded:  python s02a_stardist_cache.py --shard k --nshards N
Merge:    python s02a_stardist_cache.py --merge
Output:   D:\vhe_count_work\stardist_v2\stardist_cache.pkl
          outputs/s02_counter/stardist_per_patch.csv
"""

import argparse
import os
import pickle
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd

import countlib as cl

CACHE_DIR = os.path.join(cl.WORK_DIR, "stardist_v2")
OUT_DIR = os.path.join(_script_dir, "outputs", "s02_counter")


def rows_to_cache():
    m, _, _ = cl.load_v2_tables()
    m["split"] = cl.split_of(m, val_fold=0)
    keep = (((m.split == "train") & (m.role == "train"))
            | ((m.split == "val") & (m.role == "eval"))
            | ((m.split == "legacy_test") & (m.role == "eval")))
    # sort by core so each shard reads few cores at a time (mmap locality)
    return m[keep].sort_values(["core", "role", "y0", "x0"]).reset_index(drop=True)


def run_shard(shard, nshards, tf_threads):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = rows_to_cache()
    rows = rows.iloc[shard::nshards]
    os.makedirs(CACHE_DIR, exist_ok=True)
    part_path = os.path.join(CACHE_DIR, f"part{shard:02d}_of{nshards:02d}.pkl")
    done = {}
    if os.path.exists(part_path):
        with open(part_path, "rb") as f:
            done = pickle.load(f)
        print(f"[shard {shard}] resuming, {len(done)} already cached")
    t0 = time.time(); n_new = 0
    for _, r in rows.iterrows():
        if r.patch_uid in done:
            continue
        _, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r["size"]), channels=[2])
        rgb = cl.resize_rgb(he, int(r.render))
        (cy, cx, ar), _ = cl.detect_nuclei(rgb)
        done[r.patch_uid] = (cy, cx, ar)
        n_new += 1
        if n_new % 200 == 0:
            with open(part_path, "wb") as f:
                pickle.dump(done, f)
            print(f"[shard {shard}] {len(done)}/{len(rows)}  {(time.time() - t0) / n_new:.2f}s/patch", flush=True)
    with open(part_path, "wb") as f:
        pickle.dump(done, f)
    print(f"[shard {shard}] complete: {len(done)} patches")


def merge():
    rows = rows_to_cache()
    allp = {}
    for fn in sorted(os.listdir(CACHE_DIR)):
        if fn.startswith("part") and fn.endswith(".pkl"):
            with open(os.path.join(CACHE_DIR, fn), "rb") as f:
                allp.update(pickle.load(f))
    missing = set(rows.patch_uid) - set(allp)
    if missing:
        raise RuntimeError(f"{len(missing)} patches missing from the StarDist cache, e.g. {sorted(missing)[:3]}. "
                           "Re-run the shards (they resume).")
    with open(os.path.join(CACHE_DIR, "stardist_cache.pkl"), "wb") as f:
        pickle.dump(allp, f)
    rows["n_sd"] = [len(allp[u][2]) for u in rows.patch_uid]
    rows["med_area"] = [float(np.median(allp[u][2])) if len(allp[u][2]) else np.nan for u in rows.patch_uid]
    os.makedirs(OUT_DIR, exist_ok=True)
    rows.to_csv(os.path.join(OUT_DIR, "stardist_per_patch.csv"), index=False)
    print(rows.groupby(["split", "role"]).n_sd.describe())
    # cross-check against v2's per-core table (same protocol, eval tiling)
    ref = cl.read_csv_bom(os.path.join(cl.V2_RESULTS, "09_nuclear_density.csv"))
    ev = rows[rows.role == "eval"].groupby("core").n_sd.median().rename("median_here")
    cmp_ = ref.set_index("core")[["median"]].join(ev, how="inner")
    if len(cmp_):
        print(f"\nCross-check vs 09_nuclear_density.csv on {len(cmp_)} cores: "
              f"median |diff| {np.median(np.abs(cmp_['median'] - cmp_['median_here'])):.1f} nuclei, "
              f"r = {np.corrcoef(cmp_['median'], cmp_['median_here'])[0, 1]:.3f}")
        cmp_.to_csv(os.path.join(OUT_DIR, "stardist_crosscheck_09.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tf_threads", type=int, default=6)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        run_shard(a.shard, a.nshards, a.tf_threads)
