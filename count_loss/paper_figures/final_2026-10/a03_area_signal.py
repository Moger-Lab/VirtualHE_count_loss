"""
a03 -- Fig 6a: is the area loss's signal aligned with the count deficit?
CG+M+A (the area-loss model), 3 seeds x epochs 27-30, held-out evaluation
cores (17 cores, 1038 patches). Inference only (handoff analysis 4).

Per patch: H-positive area fraction = fraction of pixels with H >= the
calibrated nuclear threshold (calib.nuclear_h_threshold), the quantity the
area loss acts on (same definition as s01_diagnostic_model3.py): fake = the
generator's own H channel; real = rgb_to_he of the real H&E (512 render).
Fake per patch = mean over the 4 checkpoints, then over seeds.
Count ratio: StarDist fake (epochs 27-30 mean, seed mean) / real round trip.
Output: data\\area_signal_per_patch.csv (real and per-run fake H-positive fractions).
Never overwrites.
"""

import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import fcommon as F
sys.path.insert(0, F.PROJECT)

import numpy as np
import pandas as pd

import countlib as cl

if __name__ == "__main__":
    import torch
    import countnet as cn
    from s18a_smoke_inputs import heldout_rows, load_G

    out = os.path.join(F.DATA, "area_signal_per_patch.csv")
    if os.path.exists(out):
        sys.exit("exists; refusing to overwrite")
    t0 = time.perf_counter()
    pt = F.patches()
    hr = heldout_rows().set_index("patch_uid").loc[pt.patch_uid].reset_index()
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    thr = calib.nuclear_h_threshold
    res = pd.DataFrame({"patch_uid": pt.patch_uid, "core": pt.core, "bucket": pt.bucket, "stromal": pt.stromal,
                        "n_real_rt": pt.n_real_rt})
    hreal = []
    for r in hr.itertuples():
        _, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r.size))
        h = cn.v1_common.rgb_to_he(cl.resize_rgb(he, cl.RENDER), calib)[0]
        hreal.append(float((h >= thr).mean()))
    res["hpos_real"] = hreal
    print(f"real done {time.perf_counter() - t0:.0f} s; threshold {thr}", flush=True)

    dev = torch.device("cuda")
    _, cfg0 = load_G("A1_full", 30, torch.device("cpu"))
    ds = cn.V2PatchDataset(hr, cfg0, calib)
    srs = torch.stack([ds[i]["srs"] for i in range(len(ds))])
    for tag in F.MODEL["CG+M+A"]["tags"]:
        per_ep = []
        for e in F.EPS:
            G, cfg = load_G(tag, e, dev)
            assert cfg["SRS_CHANNEL_INDICES"] == cfg0["SRS_CHANNEL_INDICES"]
            v = []
            for s in range(0, len(srs), 16):
                with torch.no_grad():
                    f = G(srs[s:s + 16].to(dev)).float()
                v.append((f[:, 0] >= thr).float().mean(dim=(1, 2)).cpu().numpy())
            per_ep.append(np.concatenate(v))
            res[f"hpos_{tag}_e{e}"] = per_ep[-1]
            del G; torch.cuda.empty_cache()
            print(f"{tag} ep{e} {time.perf_counter() - t0:.0f} s", flush=True)
        res[f"hpos_{tag}"] = np.mean(per_ep, axis=0)
    res.to_csv(out, index=False)
    print("wrote", out, f"{(time.perf_counter() - t0) / 60:.1f} min")
