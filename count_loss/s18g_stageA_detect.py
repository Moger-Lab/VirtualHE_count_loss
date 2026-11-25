"""
s18g -- run ONE detector over the Stage A images and save measurement tables.

Env per detector:  stardist -> srs-analysis;  cellvit_sam -> vhe-cellvit;
                   hovernet, cellpose, instanseg -> vhe-detectors.
For each image set (real, rt, and every run folder under img\\):
  tables\\<det>\\<set>_nuclei.csv.gz   one row per nucleus: patch_uid, nid,
                                       cy, cx (512 grid px), area_um2,
                                       solidity, eccentricity, mean_H, mean_E
  tables\\<det>\\<set>_patches.csv     one row per patch: n, M1-M5 summaries,
                                       tissue_frac, H_p99_tissue, sec (detector
                                       time for this image)
Resumable per set; never overwrites a finished set.

Usage: python s18g_stageA_detect.py --det cellvit_sam [--root D:\\...\\stageA] [--sets real rt A5_full]
"""

import argparse
import glob
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd

import s18lib as sl

DEFAULT_ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageA")


def make_detector(name):
    """Returns a callable rgb(512,512,3) uint8 -> int label image (512 grid)
    and a sync function for timing."""
    if name == "stardist":
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(12)
        import countlib as cl
        return (lambda rgb: cl.detect_nuclei(rgb)[1]), (lambda: None)
    import torch
    sync = torch.cuda.synchronize
    if name == "cellvit_sam":
        import s18c_cellvit as cv
        dev = torch.device("cuda")
        m, mean, std = cv.load("SAM", dev)
        return (lambda rgb: cv.run(m, mean, std, rgb, dev)), sync
    import s18b_detectors as db
    return {"hovernet": db.HoverNet, "cellpose": db.Cellpose, "instanseg": db.InstanSegDet}[name](), sync


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["stardist", "cellvit_sam", "hovernet", "cellpose", "instanseg"])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--sets", nargs="*", default=None)
    a = ap.parse_args()

    pt = pd.read_csv(os.path.join(a.root, "patches.csv"))
    sets = a.sets or (["real", "rt"] + sorted(d for d in os.listdir(os.path.join(a.root, "img"))
                                              if d not in ("real", "rt")))
    out = os.path.join(a.root, "tables", a.det)
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    det, sync = make_detector(a.det)
    print(f"{a.det}: loaded in {time.perf_counter() - t0:.1f} s; sets {sets}", flush=True)
    warm = False
    for s in sets:
        fp = os.path.join(out, f"{s}_patches.csv")
        if os.path.exists(fp):
            print(f"  {s}: done already", flush=True); continue
        prow, ntabs = [], []
        t_set = time.perf_counter()
        for u in pt.patch_uid:
            rgb = sl.read_png(os.path.join(a.root, "img", s, f"{u}.png"))
            if not warm:
                det(rgb); warm = True
            sync(); t = time.perf_counter()
            lab = det(rgb)
            sync(); sec = time.perf_counter() - t
            rec, tab = sl.measure_image(rgb, lab)
            rec.update(patch_uid=u, sec=sec)
            prow.append(rec)
            tab.insert(0, "patch_uid", u)
            ntabs.append(tab)
        nt = pd.concat(ntabs, ignore_index=True)
        nt.to_csv(os.path.join(out, f"{s}_nuclei.csv.gz"), index=False, float_format="%.5g")
        P = pd.DataFrame(prow)
        P.to_csv(fp + ".tmp", index=False); os.replace(fp + ".tmp", fp)
        print(f"  {s}: {len(P)} patches, {len(nt)} nuclei, {time.perf_counter() - t_set:.0f} s "
              f"(detector median {P.sec.median():.3f} s/img), empty {int((P.n == 0).sum())}", flush=True)
