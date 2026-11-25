"""
a04 -- does the pre-registered contrast replicate under independent
detectors? (handoff analysis 1; supplementary S5). Descriptive: the second
detectors fail the seed-noise rule (D-32), so this is reported, not ranked.

M6 (r, log-log slope, disc) under StarDist, CellViT-SAM-H and InstanSeg:
per-patch fake count = mean over epochs 27-30 (s18 stage A ep30 tables +
stage B ep27-29 tables; StarDist from the eval CSVs), real = that
detector's round-trip count, buckets on the original-RGB StarDist count.
Model means and paired differences (by seed) with core-cluster bootstrap
CIs, indices exactly as s18k. Output: data\\detector_m6_{table,paired}.csv.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd

DETS = {"stardist": "StarDist", "cellvit_sam": "CellViT-SAM-H", "instanseg": "InstanSeg"}
STAGEB = __import__("vhe_paths").work("s18_metric_screen", "stageB_e{e}")
COMPS = [("CG+M minus CG", "CG+M", "CG", False), ("CG+M+A minus CG+M", "CG+M+A", "CG+M", False),
         ("CG+M+C minus CG+M", "CG+M+C", "CG+M", False), ("CG+M+C minus CG+M+A", "CG+M+C", "CG+M+A", True)]
KEYS = ("r", "slope", "disc")


def counts(det, tag, uids, R):
    if det == "stardist":
        return R[tag].n_fake.values
    v = []
    for e in F.EPS:
        root = F.STAGEA if e == 30 else STAGEB.format(e=e)
        p = pd.read_csv(os.path.join(root, "tables", det, f"{tag}_patches.csv"))
        assert list(p.patch_uid) == list(uids)
        v.append(p.n.values.astype(float))
    return np.mean(v, axis=0)


if __name__ == "__main__":
    outs = [os.path.join(F.DATA, f"detector_m6_{n}.csv") for n in ("table", "paired")]
    if any(os.path.exists(p) for p in outs):
        sys.exit("exists; refusing to overwrite")
    pt, R = F.load_all()
    uids = pt.patch_uid.values
    bucket = pt.bucket.values
    bidx = F.boot_indices(pt)
    tab, pair = [], []
    for det, name in DETS.items():
        nr = pd.read_csv(os.path.join(F.STAGEA, "tables", det, "rt_patches.csv")).n.values.astype(float)
        NF = {t: counts(det, t, uids, R) for k in F.KEYS for t in F.MODEL[k]["tags"]}
        pt_m = {t: F.m6(NF[t], nr, bucket) for t in NF}
        bs = [{t: F.m6(NF[t][i], nr[i], bucket[i]) for t in NF} for i in bidx]
        for k in F.KEYS:
            tags = F.MODEL[k]["tags"]
            for m in KEYS:
                b = [np.mean([x[t][m] for t in tags]) for x in bs]
                lo, hi = np.percentile(b, [2.5, 97.5])
                tab.append(dict(detector=name, model=k, metric=m, value=np.mean([pt_m[t][m] for t in tags]),
                                ci_lo=lo, ci_hi=hi, seed_values=";".join(f"{pt_m[t][m]:.3f}" for t in tags)))
        for cname, a, b_, pre in COMPS:
            for m in KEYS:
                d = lambda src: np.mean([src[ta][m] - src[tb][m] for ta, tb in zip(F.MODEL[a]["tags"], F.MODEL[b_]["tags"])])
                bb = [d(x) for x in bs]
                lo, hi = np.percentile(bb, [2.5, 97.5])
                pair.append(dict(detector=name, comparison=cname, a=a, b=b_, pre_registered=pre, metric=m,
                                 diff=d(pt_m), ci_lo=lo, ci_hi=hi, excl_0=bool(lo > 0 or hi < 0)))
        print("done", name, flush=True)
    T, P = pd.DataFrame(tab), pd.DataFrame(pair)
    T.to_csv(outs[0], index=False); P.to_csv(outs[1], index=False)
    # check: StarDist and second-detector means equal the s18 stage B table
    sb = pd.read_csv(os.path.join(F.PROJECT, "outputs", "s18_metric_screen", "stageB", "stageB_table.csv"))
    dev = 0.0
    for r in sb.itertuples():
        for k in F.KEYS:
            v = T[(T.detector == DETS[r.detector]) & (T.model == k) & (T.metric == r.metric.split()[-1])].value.iloc[0]
            dev = max(dev, abs(v - getattr(r, k.replace("+", "_")) if hasattr(r, k.replace("+", "_")) else 0))
    sbv = sb.set_index(["detector", "metric"])
    dev = max(abs(T[(T.detector == DETS[d]) & (T.model == k) & (T.metric == m)].value.iloc[0] - sbv.loc[(d, f"M6 density {m}"), k])
              for d in DETS for k in F.KEYS for m in KEYS)
    print(f"max |diff| vs stageB_table: {dev:.2e}")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(P[P.pre_registered].to_string(index=False))
