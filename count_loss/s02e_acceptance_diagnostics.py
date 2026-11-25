"""
s02e -- diagnostics for the two acceptance items that did not pass cleanly.

C'. Synthetic test with an empty-canvas baseline: every synthetic canvas
    read ~+1.1 over its blob count. Measure the empty canvas (with and
    without the added noise) and report baseline-subtracted counts.
B'. Why the counter under-reads Model 3 fakes in the LOW bucket: StarDist
    percentile-normalises each RGB image (1 / 99.8), so it is blind to
    absolute stain; the counter reads absolute calibrated H. Test whether
    the counter/StarDist ratio on fakes tracks how pale the fake H is.
"""

import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

import countlib as cl
import countnet as cn

OUT_DIR = os.path.join(_script_dir, "outputs", "s02_counter")

if __name__ == "__main__":
    dev = torch.device("cuda")
    net = cn.load_counter(os.path.join(cl.WORK_DIR, "counter", "counter_best.pt"), dev)
    acc = json.load(open(os.path.join(OUT_DIR, "acceptance.json")))
    h_nuc, h_bg, r = acc["C"]["h_nuc"], acc["C"]["h_bg"], acc["C"]["radius"]
    out = {}

    # ---- C' ---------------------------------------------------------------
    def cnt(h):
        with torch.no_grad():
            return net(torch.from_numpy(h.astype(np.float32))[None, None].to(dev)).sum().item()
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    for noise in (0.0, 0.02):
        rng = np.random.default_rng(0)
        base = np.clip(h_bg + rng.normal(0, noise, (512, 512)), -1, 1)
        e = cnt(base)
        res = {"empty": e}
        for name, blobs in {"one": [(0, 0, r)], "two_touching": [(0, -r, r), (0, r, r)],
                            "one_large_equal_area": [(0, 0, r * np.sqrt(2))],
                            "two_overlapping_20pct": [(0, -0.8 * r, r), (0, 0.8 * r, r)]}.items():
            vals = []
            for cy, cx in [(y, x) for y in (96, 256, 416) for x in (96, 256, 416)]:
                m = np.zeros((512, 512), np.float32)
                for (dy, dx, rad) in blobs:
                    m = np.maximum(m, (((yy - cy - dy) ** 2 + (xx - cx - dx) ** 2) <= rad ** 2).astype(np.float32))
                h = np.clip(base + (h_nuc - h_bg) * gaussian_filter(m, 1.5), -1, 1)
                vals.append(cnt(h) - e)
            res[name] = float(np.mean(vals))
        out[f"synthetic_noise_{noise}"] = res
        print(f"C' noise {noise}: empty canvas {e:.2f} | baseline-subtracted: " +
              ", ".join(f"{k} {v:.2f}" for k, v in res.items() if k != "empty"))

    # ---- B' ---------------------------------------------------------------
    per = pd.read_csv(os.path.join(OUT_DIR, "acceptance_model3_per_patch.csv"))
    cache = os.path.join(cl.WORK_DIR, "s01_model3_fakes")
    fh_mean, fh_p99, rh_p99 = [], [], []
    for pid in per.patch_id:
        z = np.load(os.path.join(cache, f"{pid}.npz"))
        f = z["fake_h"].astype(np.float32); rr = z["real_h"].astype(np.float32)
        fh_mean.append(f.mean()); fh_p99.append(np.percentile(f, 99)); rh_p99.append(np.percentile(rr, 99))
    per["fake_h_p99"] = fh_p99; per["real_h_p99"] = rh_p99
    per["ct_sd_ratio"] = per.counter_fake / per.n_fake.clip(lower=1)
    per["bucket"] = cl.density_bucket(per.n_real.values)
    print("\nB' counter/StarDist on fakes versus fake-H darkness (p99 of fake H):")
    for b in cl.BUCKET_NAMES:
        d = per[(per.bucket == b) & (per.n_fake >= 3)]
        rho, p = spearmanr(d.fake_h_p99, d.ct_sd_ratio)
        q = pd.qcut(d.fake_h_p99, 3, labels=["palest", "mid", "darkest"])
        by = d.groupby(q, observed=True).apply(lambda g: g.counter_fake.sum() / g.n_fake.sum())
        print(f"  {b:4s}: fake-H p99 median {d.fake_h_p99.median():+.3f} (real {d.real_h_p99.median():+.3f}); "
              f"Spearman(p99, ratio) {rho:+.3f} p={p:.1e}; ratio of sums by darkness tercile "
              + " / ".join(f"{k} {v:.3f}" for k, v in by.items()))
        out[f"B_{b}"] = dict(rho=rho, p=p, fake_p99=d.fake_h_p99.median(), real_p99=d.real_h_p99.median(),
                             by_tercile={str(k): float(v) for k, v in by.items()})
    json.dump(out, open(os.path.join(OUT_DIR, "acceptance_diagnostics.json"), "w"), indent=1, default=float)
