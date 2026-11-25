"""
g03 -- round-core versions of the Fig 4d / S10 mosaics (srs-analysis env, GPU).

g02 tiled only the evaluation patches, which leaves ragged, notched outlines.
Here every 441 px (100 um) grid position that touches the core disc is
generated / read (the evaluation patches are included at their exact
positions, same generation path and seed 42 / epoch 30 as g02), the real core
disc is fitted from the real H&E, and everything outside the disc is white.

Checks: fake counts on evaluation patches equal the evaluation CSV n_fake
(508/508 as in g02); real StarDist counts (this script's own run on the
resized original RGB) equal patches.csv n_real on evaluation patches.
Patches outside the evaluation set (the rim) are NOT in any evaluation number;
they are flagged in_eval = False in the output.

The last grid row/column starts at 3969 but the array ends at 4408, so those
patches are read at 3967 and shifted by the 2 native px (placed exactly).

Outputs: img_cores\\<core>_<set>_full.png (never overwritten),
data\\fig4d_nuclei_full.csv, data\\fig4d_circle.csv, data\\fig4d_cores_full.csv.
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
from g02_core_nuclei import IMGC, TAGS, R, choose_cores

P = 441
N = 10
ARR = 4408


def fit_circle(core):
    from scipy.ndimage import gaussian_filter, binary_closing
    _, he = cl.core_arrays(core)
    a = np.asarray(he[::8, ::8]).astype(np.float32)
    od = -np.log10((a + 1.0) / 256.0).sum(-1)
    m = binary_closing(gaussian_filter(od, 1.0) > 0.20, iterations=3)
    ys, xs = np.nonzero(m)
    x0, x1 = np.percentile(xs, [0.05, 99.95]) * 8
    y0, y1 = np.percentile(ys, [0.05, 99.95]) * 8
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = ((x1 - x0) + (y1 - y0)) / 4 * 1.03
    return cx, cy, r


def grid_positions(cx, cy, r):
    out = []
    for gr in range(N):
        for gc in range(N):
            y0, x0 = min(gr * P, ARR - P), min(gc * P, ARR - P)
            ty, tx = gr * P, gc * P  # nominal tile origin (native)
            dy = np.clip(cy, ty, ty + P) - cy
            dx = np.clip(cx, tx, tx + P) - cx
            if dx * dx + dy * dy < r * r:
                out.append(dict(row=gr, col=gc, y0=y0, x0=x0, shy=int(round((ty - y0) * R / P)), shx=int(round((tx - x0) * R / P))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    import torch
    import countnet as cn
    from s18a_smoke_inputs import load_G

    for p in ("fig4d_nuclei_full.csv", "fig4d_circle.csv", "fig4d_cores_full.csv"):
        if os.path.exists(os.path.join(F.DATA, p)):
            sys.exit(f"{p} exists; refusing to overwrite")
    pt = pd.read_csv(os.path.join(F.STAGEA, "patches.csv"))
    sel, _ = choose_cores(pt)
    sc = cl.stardist_cache()
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    dev = torch.device("cuda")
    _, cfg0 = load_G(TAGS[0][1], 30, torch.device("cpu"))

    from s18a_smoke_inputs import heldout_rows
    ho = heldout_rows().set_index("patch_uid")

    circ, cents, rows_out = [], [], []
    for rule, core in sel:
        cx, cy, r = fit_circle(core)
        circ.append(dict(core=core, cx_native=cx, cy_native=cy, r_native=r))
        print(core, "circle (native px)", round(cx), round(cy), round(r), flush=True)
        G = grid_positions(cx, cy, r)
        ev = pt[pt.core == core].copy()
        ev["row"] = (ev.y0 // P).astype(int); ev["col"] = (ev.x0 // P).astype(int)
        key = {(a, b): u for a, b, u in zip(ev.row, ev.col, ev.patch_uid)}
        assert all((int(a), int(b)) in set(zip(G.row, G.col)) for a, b in zip(ev.row, ev.col)), "an evaluation patch lies outside the fitted disc"
        G["patch_uid"] = [key.get((int(a), int(b)), f"{core}_full_{y}_{x}") for a, b, y, x in zip(G.row, G.col, G.y0, G.x0)]
        G["in_eval"] = [(int(a), int(b)) in key for a, b in zip(G.row, G.col)]
        G["core"] = core; G["size"] = P; G["render"] = R; G["confidence_weight"] = 1.0; G["n_real"] = np.nan
        print(core, "tiles", len(G), "evaluation", int(G.in_eval.sum()), flush=True)
        # evaluation patches must sit at their evaluation coordinates
        assert (G[G.in_eval].set_index("patch_uid").loc[ev.patch_uid, ["y0", "x0"]].values == ev[["y0", "x0"]].values).all()

        real = {}
        for g in G.itertuples():
            _, he = cl.v2_patch_native(core, int(g.y0), int(g.x0), P)
            real[g.patch_uid] = cl.resize_rgb(he, R)
        ds = cn.V2PatchDataset(G, cfg0, calib)
        srs = torch.stack([ds[i]["srs"] for i in range(len(ds))])
        imgs = {"real": real}
        for name, tag in TAGS:
            Gn, cfg = load_G(tag, 30, dev)
            assert cfg["SRS_CHANNEL_INDICES"] == cfg0["SRS_CHANNEL_INDICES"] and cfg["SRS_NORM_MODE"] == cfg0["SRS_NORM_MODE"]
            imgs[name] = {}
            for s in range(0, len(G), 8):
                idx = list(range(s, min(s + 8, len(G))))
                with torch.no_grad():
                    f = Gn(srs[idx].to(dev)).float().cpu().numpy()
                for j, i in enumerate(idx):
                    imgs[name][G.patch_uid[i]] = cn.v1_common.he_to_rgb(f[j, 0], f[j, 1], calib)
            del Gn; torch.cuda.empty_cache()
            print("generated", core, name, flush=True)

        for name in ["real"] + [n for n, _ in TAGS]:
            for g in G.itertuples():
                (cyy, cxx, _), _ = cl.detect_nuclei(imgs[name][g.patch_uid])
                cents += [dict(set=name, core=core, patch_uid=g.patch_uid, row=g.row, col=g.col, in_eval=g.in_eval,
                               gy=g.row * R + a - g.shy, gx=g.col * R + b - g.shx)
                          for a, b in zip(np.asarray(cyy), np.asarray(cxx))]
            print("detected", core, name, flush=True)

        # mosaics, white outside the disc
        S = R / P
        yy, xx = np.mgrid[0:N * R, 0:N * R]
        inside = (xx - cx * S) ** 2 + (yy - cy * S) ** 2 <= (r * S) ** 2
        for name in ["real"] + [n for n, _ in TAGS]:
            M = np.full((N * R, N * R, 3), 255, np.uint8)
            for g in G.itertuples():
                t = imgs[name][g.patch_uid]
                M[g.row * R:g.row * R + R - g.shy, g.col * R:g.col * R + R - g.shx] = t[g.shy:, g.shx:]
            M[~inside] = 255
            p = os.path.join(IMGC, f"{core}_{name}_full.png")
            if not os.path.exists(p):
                sl.write_png(p, M)
        G.to_csv(os.path.join(F.DATA, f"fig4d_tiles_{core}.csv"), index=False)

    C = pd.DataFrame(cents)
    cc = pd.DataFrame(circ)
    C = C.merge(cc, on="core")
    S = R / P
    d2 = (C.gx - C.cx_native * S) ** 2 + (C.gy - C.cy_native * S) ** 2
    C["in_disc"] = d2 <= (C.r_native * S) ** 2
    C = C.drop(columns=["cx_native", "cy_native", "r_native"])

    # checks
    bad = 0
    for name, tag in TAGS:
        e = pd.concat([pd.read_csv(os.path.join(F.RUNS_DIR, tag, f"G_S2H_epoch030_{s}_eval.csv")) for s in ("val", "legacy_test")]).set_index("patch_uid")
        for _, core in sel:
            u = pt[pt.core == core].patch_uid
            n = C[(C.set == name) & C.in_eval & (C.core == core)].groupby("patch_uid").size().reindex(u).fillna(0).astype(int)
            bad += int((n.values != e.loc[u, "n_fake"].values).sum())
    print("fake count mismatches vs eval CSV:", bad, "of", len(pt[pt.core.isin([c for _, c in sel])]) * len(TAGS))
    badr = 0
    for _, core in sel:
        u = pt[pt.core == core].set_index("patch_uid")
        n = C[(C.set == "real") & C.in_eval & (C.core == core)].groupby("patch_uid").size().reindex(u.index).fillna(0).astype(int)
        badr += int((n.values != u.n_real.values).sum())
    print("real count mismatches vs patches.csv n_real:", badr)

    rows = []
    for rule, core in sel:
        for name in ["real"] + [n for n, _ in TAGS]:
            X = C[(C.set == name) & (C.core == core)]
            rows.append(dict(rule=rule, core=core, set=name, n_in_disc=int(X.in_disc.sum()),
                             n_eval_patches_only=int(X.in_eval.sum())))
    C.to_csv(os.path.join(F.DATA, "fig4d_nuclei_full.csv"), index=False)
    cc.to_csv(os.path.join(F.DATA, "fig4d_circle.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(F.DATA, "fig4d_cores_full.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
