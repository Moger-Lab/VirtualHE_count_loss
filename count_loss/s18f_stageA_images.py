"""
s18f -- Stage A images for the s18 screen (srs-analysis env).

Writes RGB PNGs (lossless) for the pooled held-out eval patches (val fold 0
+ legacy test, 17 cores, 1038 patches):
  img\\real\\<uid>.png   real H&E, exactly as the StarDist cache saw it
  img\\rt\\<uid>.png     he_to_rgb(rgb_to_he(real))   (R1 round trip)
  img\\<tag>\\<uid>.png  he_to_rgb(G(srs)) for each of the 12 paper runs
at one epoch (default 30), plus patches.csv (density class, stromal flag,
gallery flag). The PNGs are TEMPORARY transfer files between the three
environments; s18j deletes them after Stage A except the gallery patches.
No H/E arrays are saved.

Usage: python s18f_stageA_images.py [--epoch 30] [--out stageA] [--limit N]
"""

import argparse
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd

import countlib as cl
import s18lib as sl
from s18a_smoke_inputs import heldout_rows, pick_stratified

RUNS = ["CG_full", "CG_full_s7", "CG_full_s123", "CGM_full", "CGM_full_s7", "CGM_full_s123",
        "A1_full", "A1_full_s7", "A1_full_s123", "A5_full", "A5_full_s7", "A5_full_s123"]
STROMAL_PCT = 10          # stromal patch: real StarDist count <= 10th percentile ...
STROMAL_MIN_TISSUE = 0.5  # ... and real tissue fraction >= 0.5 (D-29)
N_GALLERY = (14, 13, 13)  # low / med / high


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--out", default="stageA")
    ap.add_argument("--limit", type=int, default=0, help="smoke: this many patches (stratified)")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import torch
    import countnet as cn
    from s18a_smoke_inputs import load_G

    root = os.path.join(cl.WORK_DIR, "s18_metric_screen", a.out)
    img = os.path.join(root, "img")
    os.makedirs(img, exist_ok=True)
    dev = torch.device("cuda")
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)

    rows = heldout_rows()
    rows["bucket"] = cl.density_bucket(rows.n_real.values)
    if a.limit:
        k = a.limit // 3
        rows = pick_stratified(rows, (a.limit - 2 * k, k, k), 1)
    rows = rows.reset_index(drop=True)
    gal = pick_stratified(rows, N_GALLERY, 2) if not a.limit else rows
    rows["gallery"] = rows.patch_uid.isin(set(gal.patch_uid)).astype(int)
    print(f"{len(rows)} patches, {rows.core.nunique()} cores, gallery {rows.gallery.sum()}", flush=True)

    # real + round trip; tissue fraction for the stromal flag
    t0 = time.perf_counter()
    for d in ("real", "rt"):
        os.makedirs(os.path.join(img, d), exist_ok=True)
    tissue = []
    for r in rows.itertuples():
        _, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r.size))
        real = cl.resize_rgb(he, cl.RENDER)
        tissue.append(float(sl.tissue_mask(real).mean()))
        p = os.path.join(img, "real", f"{r.patch_uid}.png")
        if not os.path.exists(p):
            sl.write_png(p, real)
        p = os.path.join(img, "rt", f"{r.patch_uid}.png")
        if not os.path.exists(p):
            h = cn.v1_common.rgb_to_he(real, calib)
            sl.write_png(p, cn.v1_common.he_to_rgb(h[0], h[1], calib))
    rows["real_tissue_frac"] = tissue
    thr = np.percentile(rows.n_real, STROMAL_PCT) if not a.limit else np.percentile(heldout_rows().n_real, STROMAL_PCT)
    rows["stromal"] = ((rows.n_real <= thr) & (rows.real_tissue_frac >= STROMAL_MIN_TISSUE)).astype(int)
    keep = ["patch_uid", "core", "split", "y0", "x0", "size", "n_real", "bucket", "real_tissue_frac", "stromal", "gallery"]
    pp = os.path.join(root, "patches.csv")
    if not os.path.exists(pp):
        rows[keep].to_csv(pp, index=False)
    print(f"real + rt done in {time.perf_counter() - t0:.0f} s; stromal threshold n_real <= {thr:.0f}, "
          f"{rows.stromal.sum()} stromal patches", flush=True)

    # SRS inputs once (all 12 runs share the SRS settings; asserted)
    cfgs = {}
    for tag in RUNS:
        G, cfg = load_G(tag, a.epoch, torch.device("cpu"))
        cfgs[tag] = cfg; del G
    keyset = {(tuple(c["SRS_CHANNEL_INDICES"]), c["SRS_NORM_MODE"], c.get("TYPE_CHANNEL_NORM", "global"),
               int(c["NETWORK_INPUT_SIZE"])) for c in cfgs.values()}
    assert len(keyset) == 1, f"runs differ in SRS input settings: {keyset}"
    ds = cn.V2PatchDataset(rows, cfgs[RUNS[0]], calib)
    t0 = time.perf_counter()
    srs = torch.stack([ds[i]["srs"] for i in range(len(ds))])
    print(f"SRS inputs {tuple(srs.shape)} in {time.perf_counter() - t0:.0f} s", flush=True)

    timing = []
    for tag in RUNS:
        od = os.path.join(img, tag); os.makedirs(od, exist_ok=True)
        todo = [i for i, u in enumerate(rows.patch_uid) if not os.path.exists(os.path.join(od, f"{u}.png"))]
        if not todo:
            continue
        G, _ = load_G(tag, a.epoch, dev)
        t0 = time.perf_counter()
        for s in range(0, len(todo), a.batch):
            idx = todo[s:s + a.batch]
            with torch.no_grad():
                f = G(srs[idx].to(dev)).float().cpu().numpy()
            for j, i in enumerate(idx):
                sl.write_png(os.path.join(od, f"{rows.patch_uid[i]}.png"), cn.v1_common.he_to_rgb(f[j, 0], f[j, 1], calib))
        dt = time.perf_counter() - t0
        timing.append(dict(run=tag, epoch=a.epoch, n=len(todo), sec=dt))
        print(f"{tag} ep{a.epoch}: {len(todo)} fakes in {dt:.0f} s", flush=True)
        del G; torch.cuda.empty_cache()
    if timing:
        pd.DataFrame(timing).to_csv(os.path.join(root, f"timing_generate_{int(time.time())}.csv"), index=False)
