"""
g02 -- whole-core images and nucleus centroids for the Fig 4 spatial panel
(srs-analysis env, GPU).

Cores (fixed rule, from the REAL section only): the densest held-out core
(highest median real StarDist count per patch) and the "typical" core (median
of the 17 core medians). Every evaluation patch of those cores is tiled in
place (non-overlapping 441 px native patches, rendered 512 px = 100 um).

For each core: real H&E mosaic (original RGB, as the StarDist cache saw it)
and fake mosaics for the four models (seed 42, epoch 30, the same generation
path as g01 / s18f); StarDist (v1 protocol, countlib.detect_nuclei) is run on
every fake patch and the cached real centroids are used for the real image.
Check: fake counts equal the evaluation CSV n_fake.

Outputs: D:\\vhe_count_work\\paper_figs_2026-10\\img_cores\\<core>_<set>.png (mosaics,
never overwritten) and data\\fig4d_nuclei.csv (set, core, patch_uid, row, col,
cy, cx on the 512 px patch grid), data\\fig4d_cores.csv (rule + counts).
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import fcommon as F
sys.path.insert(0, F.PROJECT)

import numpy as np
import pandas as pd

import countlib as cl
import s18lib as sl

IMGC = __import__("vhe_paths").work("paper_figs_2026-10", "img_cores")
TAGS = [("CG", "CG_full"), ("CG+M", "CGM_full"), ("CG+M+A", "A1_full"), ("CG+M+C", "A5_full")]
R = 512


def choose_cores(pt):
    med = pt.groupby("core").n_real.median().sort_values()
    dense = med.index[-1]
    typical = med.index[len(med) // 2]
    return [("densest", dense), ("typical", typical)], med


if __name__ == "__main__":
    import torch
    import countnet as cn
    from s18a_smoke_inputs import heldout_rows, load_G

    for p in ("fig4d_nuclei.csv", "fig4d_cores.csv"):
        if os.path.exists(os.path.join(F.DATA, p)):
            sys.exit(f"{p} exists; refusing to overwrite")
    os.makedirs(IMGC, exist_ok=True)
    pt = pd.read_csv(os.path.join(F.STAGEA, "patches.csv"))
    sel, med = choose_cores(pt)
    print("core medians:", med.to_dict()); print("selected:", sel)
    pool = pt[pt.core.isin([c for _, c in sel])].copy().reset_index(drop=True)
    pool["row"] = (pool.y0 // 441).astype(int); pool["col"] = (pool.x0 // 441).astype(int)
    assert (pool.y0 % 441 == 0).all() and (pool.x0 % 441 == 0).all() and (pool["size"] == 441).all()

    sc = cl.stardist_cache()
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    real = {}
    for r in pool.itertuples():
        _, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r.size))
        real[r.patch_uid] = cl.resize_rgb(he, cl.RENDER)
    hr = heldout_rows().set_index("patch_uid").loc[pool.patch_uid].reset_index()
    dev = torch.device("cuda")
    _, cfg0 = load_G(TAGS[0][1], 30, torch.device("cpu"))
    ds = cn.V2PatchDataset(hr, cfg0, calib)
    srs = torch.stack([ds[i]["srs"] for i in range(len(ds))])

    imgs = {"real": real}
    cents = []
    for r in pool.itertuples():
        cy, cx = np.asarray(sc[r.patch_uid][0]), np.asarray(sc[r.patch_uid][1])
        cents += [dict(set="real", core=r.core, patch_uid=r.patch_uid, row=r.row, col=r.col, cy=a, cx=b) for a, b in zip(cy, cx)]
    for name, tag in TAGS:
        G, cfg = load_G(tag, 30, dev)
        assert cfg["SRS_CHANNEL_INDICES"] == cfg0["SRS_CHANNEL_INDICES"] and cfg["SRS_NORM_MODE"] == cfg0["SRS_NORM_MODE"]
        imgs[name] = {}
        for s in range(0, len(hr), 8):
            idx = list(range(s, min(s + 8, len(hr))))
            with torch.no_grad():
                f = G(srs[idx].to(dev)).float().cpu().numpy()
            for j, i in enumerate(idx):
                imgs[name][hr.patch_uid[i]] = cn.v1_common.he_to_rgb(f[j, 0], f[j, 1], calib)
        del G; torch.cuda.empty_cache()
        for r in pool.itertuples():
            (cy, cx, _), _ = cl.detect_nuclei(imgs[name][r.patch_uid])
            cents += [dict(set=name, core=r.core, patch_uid=r.patch_uid, row=r.row, col=r.col, cy=a, cx=b) for a, b in zip(cy, cx)]
        print("done", name, flush=True)

    C = pd.DataFrame(cents)
    bad = 0
    for name, tag in TAGS:
        e = pd.concat([pd.read_csv(os.path.join(F.RUNS_DIR, tag, f"G_S2H_epoch030_{s}_eval.csv")) for s in ("val", "legacy_test")]).set_index("patch_uid")
        n = C[C.set == name].groupby("patch_uid").size().reindex(pool.patch_uid).fillna(0).astype(int)
        bad += int((n.values != e.loc[pool.patch_uid, "n_fake"].values).sum())
    print("fake count mismatches vs eval CSV:", bad, "of", len(pool) * len(TAGS))
    real_n = C[C.set == "real"].groupby("patch_uid").size().reindex(pool.patch_uid).fillna(0).astype(int)
    print("real count mismatches vs patches.csv n_real:", int((real_n.values != pool.n_real.values).sum()))

    rows = []
    for rule, core in sel:
        P = pool[pool.core == core]
        nr, nc = P.row.max() + 1, P.col.max() + 1
        for name in ["real"] + [n for n, _ in TAGS]:
            M = np.full((nr * R, nc * R, 3), 255, np.uint8)
            for r in P.itertuples():
                M[r.row * R:(r.row + 1) * R, r.col * R:(r.col + 1) * R] = imgs[name][r.patch_uid]
            p = os.path.join(IMGC, f"{core}_{name}.png")
            if not os.path.exists(p):
                sl.write_png(p, M)
            rows.append(dict(rule=rule, core=core, set=name, n_patches=len(P), grid_rows=nr, grid_cols=nc,
                             n_nuclei=int(((C.set == name) & (C.core == core)).sum())))
    C.to_csv(os.path.join(F.DATA, "fig4d_nuclei.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(F.DATA, "fig4d_cores.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
