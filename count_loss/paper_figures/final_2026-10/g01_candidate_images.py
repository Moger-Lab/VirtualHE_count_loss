"""
g01 -- images and rule-based candidate selection for the example-image
figures (Fig 2 and S6; handoff 2026-10-02 section 6). srs-analysis env, GPU.

Pool: the 40-patch stratified gallery of s18 stage A (14 / 13 / 13 sparse /
intermediate / dense, 6 of the sparse are stromal) plus C-3_eval_1764_2646
(owner request: real 100 vs fakes 29-46).

For every pool patch, writes PNGs to D:\\vhe_count_work\\paper_figs_2026-10\\img\\<set>\\<uid>.png:
  real   real H&E exactly as StarDist saw it (resize of the native patch)
  srs    SRS input (channel 1) for display: core-level 0.5-99.5 percentile stretch
  <tag>  he_to_rgb(G(srs)) at epoch 30 for the seed-42 runs of the four
         models and the seed-7 / seed-123 CG runs (S6); same code path as s18f.

Selection (rule-based, from the REAL section only; never from model output):
  eligible   valid (registered) fraction of the patch = 1, real tissue
             fraction >= 0.5, dark-artefact fraction (summed OD > 1.6) < 2%
  quality    rank sum of focus (variance of Laplacian of real grey) and
             SRS-H&E tissue-outline agreement (IoU of the two tissue masks;
             proxy for no visible registration offset)
  versions   A-D: in each row class (sparse non-stromal, intermediate, dense,
             stromal) eligible patches in quality order are dealt to A, B, C,
             D, skipping a patch whose core is already in that version; the
             C-3 patch is forced into the dense row of version D.
  unselected (E): one patch per row class drawn with numpy seed 0 from the
             whole pool, no quality rule (supplementary counterpart).
Outputs: data\\fig2_pool_quality.csv, data\\fig2_candidates.csv. Never overwrites a PNG.
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

IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
FORCED = "C-3_eval_1764_2646"
RUNS = ["CG_full", "CGM_full", "A1_full", "A5_full", "CG_full_s7", "CG_full_s123"]
ROWS = ["sparse", "intermediate", "dense", "stromal"]


def row_class(r):
    if r.stromal == 1:
        return "stromal"
    return {"low": "sparse", "med": "intermediate", "high": "dense"}[r.bucket]


_core_pct = {}


def srs_display(core, y0, x0, size):
    srs, _ = cl.core_arrays(core)
    if core not in _core_pct:
        valid = np.load(os.path.join(cl.ALIGNED_DIR, f"{core}_valid.npy"), mmap_mode="r")
        s = np.asarray(srs[1, ::8, ::8]).astype(np.float32)
        _core_pct[core] = np.percentile(s[np.asarray(valid[::8, ::8])], [0.5, 99.5])
    lo, hi = _core_pct[core]
    p = np.asarray(srs[1, y0:y0 + size, x0:x0 + size]).astype(np.float32)
    g = np.clip((p - lo) / (hi - lo), 0, 1)
    g = cl.resize_chw(g[None], cl.RENDER)[0]
    return (np.repeat(g[..., None], 3, -1) * 255).astype(np.uint8)


def quality(r, real, srs_rgb):
    from scipy.ndimage import laplace
    from skimage.filters import threshold_otsu
    valid = np.load(os.path.join(cl.ALIGNED_DIR, f"{r.core}_valid.npy"), mmap_mode="r")
    vf = float(np.asarray(valid[r.y0:r.y0 + r.size, r.x0:r.x0 + r.size]).mean())
    grey = real.astype(np.float32).mean(-1) / 255.0
    focus = float(laplace(grey).var())
    od = -np.log10((real.astype(np.float32) + 1.0) / 256.0).sum(-1)
    dark = float((od > 1.6).mean())
    t_he = sl.tissue_mask(real)
    s = srs_rgb[..., 0].astype(np.float32)
    t_srs = s > threshold_otsu(s) if s.std() > 0 else np.zeros_like(t_he)
    iou = float((t_he & t_srs).sum() / max((t_he | t_srs).sum(), 1))
    return dict(valid_frac=vf, focus=focus, dark_frac=dark, tissue_iou=iou)


if __name__ == "__main__":
    import torch
    import countnet as cn
    from s18a_smoke_inputs import heldout_rows, load_G

    for p in ("fig2_pool_quality.csv", "fig2_candidates.csv"):
        if os.path.exists(os.path.join(F.DATA, p)):
            sys.exit(f"{p} exists; refusing to overwrite")
    pt = pd.read_csv(os.path.join(F.STAGEA, "patches.csv"))
    pool = pt[(pt.gallery == 1) | (pt.patch_uid == FORCED)].copy().reset_index(drop=True)
    pool["row"] = [row_class(r) for r in pool.itertuples()]
    print(f"pool {len(pool)} patches:", pool.row.value_counts().to_dict())

    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    for d in ["real", "srs"] + RUNS:
        os.makedirs(os.path.join(IMG, d), exist_ok=True)
    q = []
    for r in pool.itertuples():
        _, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r.size))
        real = cl.resize_rgb(he, cl.RENDER)
        srs_rgb = srs_display(r.core, int(r.y0), int(r.x0), int(r.size))
        for d, im in (("real", real), ("srs", srs_rgb)):
            p = os.path.join(IMG, d, f"{r.patch_uid}.png")
            if not os.path.exists(p):
                sl.write_png(p, im)
        q.append(quality(r, real, srs_rgb))
    pool = pd.concat([pool, pd.DataFrame(q)], axis=1)

    # generator outputs, same path as s18f
    hr = heldout_rows().set_index("patch_uid").loc[pool.patch_uid].reset_index()
    dev = torch.device("cuda")
    _, cfg0 = load_G(RUNS[0], 30, torch.device("cpu"))
    ds = cn.V2PatchDataset(hr, cfg0, calib)
    srs = torch.stack([ds[i]["srs"] for i in range(len(ds))])
    for tag in RUNS:
        G, cfg = load_G(tag, 30, dev)
        assert cfg["SRS_CHANNEL_INDICES"] == cfg0["SRS_CHANNEL_INDICES"] and cfg["SRS_NORM_MODE"] == cfg0["SRS_NORM_MODE"]
        for s in range(0, len(hr), 8):
            idx = list(range(s, min(s + 8, len(hr))))
            todo = [i for i in idx if not os.path.exists(os.path.join(IMG, tag, f"{hr.patch_uid[i]}.png"))]
            if not todo:
                continue
            with torch.no_grad():
                f = G(srs[todo].to(dev)).float().cpu().numpy()
            for j, i in enumerate(todo):
                sl.write_png(os.path.join(IMG, tag, f"{hr.patch_uid[i]}.png"), cn.v1_common.he_to_rgb(f[j, 0], f[j, 1], calib))
        del G; torch.cuda.empty_cache()
        print("generated", tag, flush=True)

    # check: gallery fakes regenerated here equal the s18 stage A gallery PNGs
    mism = 0
    for tag in RUNS:
        for u in pool.patch_uid:
            g = os.path.join(F.STAGEA, "gallery", tag, f"{u}.png")
            if os.path.exists(g):
                mism += int(np.abs(sl.read_png(g).astype(int) - sl.read_png(os.path.join(IMG, tag, f"{u}.png")).astype(int)).max() > 2)
    print("PNG mismatches vs s18 gallery (max abs diff > 2):", mism)

    # counts: real (original RGB, StarDist cache) and seed-42 / CG-seed fakes at epoch 30 (eval CSVs)
    for tag in RUNS:
        e = pd.concat([pd.read_csv(os.path.join(F.RUNS_DIR, tag, f"G_S2H_epoch030_{s}_eval.csv"))
                       for s in ("val", "legacy_test")]).set_index("patch_uid")
        pool[f"n_{tag}"] = e.loc[pool.patch_uid, "n_fake"].values

    pool["eligible"] = (pool.valid_frac >= 0.999) & (pool.real_tissue_frac >= 0.5) & (pool.dark_frac < 0.02)
    pool["quality_rank"] = (pool.focus.rank(ascending=False) + pool.tissue_iou.rank(ascending=False))
    pool.to_csv(os.path.join(F.DATA, "fig2_pool_quality.csv"), index=False)
    print(pool.groupby("row").eligible.agg(["sum", "count"]).to_string())

    vers = {v: [] for v in "ABCD"}
    for row in ROWS:
        cand = pool[(pool.row == row) & pool.eligible & (pool.patch_uid != FORCED)].sort_values("quality_rank")
        if row == "stromal" and len(cand) < 4:  # too few: relax to all stromal pool patches, flagged
            cand = pool[(pool.row == row)].sort_values(["eligible", "quality_rank"], ascending=[False, True])
        order = list("ABCD")
        if row == "dense":
            vers["D"].append(dict(version="D", row=row, patch_uid=FORCED, rule="forced (owner: C-3 example)"))
            order = list("ABC")
        used = set()
        for v in order:
            cores_v = {pool.set_index("patch_uid").loc[x["patch_uid"], "core"] for x in vers[v]}
            pick = next((c for c in cand.itertuples() if c.patch_uid not in used and c.core not in cores_v), None)
            if pick is None:
                pick = next(c for c in cand.itertuples() if c.patch_uid not in used)
            used.add(pick.patch_uid)
            vers[v].append(dict(version=v, row=row, patch_uid=pick.patch_uid,
                                rule="quality order" + ("" if pick.eligible else " (relaxed: not all quality rules met)")))
    rng = np.random.default_rng(0)
    E = [dict(version="E (unselected)", row=row, patch_uid=pool[(pool.row == row) & (pool.patch_uid != FORCED)]
              .patch_uid.values[rng.integers((pool.row == row).sum() - (1 if row == "dense" else 0))],
              rule="random, seed 0") for row in ROWS]
    C = pd.DataFrame([x for v in "ABCD" for x in vers[v]] + E)
    C = C.merge(pool[["patch_uid", "core", "bucket", "stromal", "n_real", "eligible", "focus", "tissue_iou",
                      "dark_frac", "valid_frac"] + [f"n_{t}" for t in RUNS]], on="patch_uid", how="left")
    C = C.rename(columns={"n_CG_full": "n_fake_CG", "n_CGM_full": "n_fake_CG+M", "n_A1_full": "n_fake_CG+M+A",
                          "n_A5_full": "n_fake_CG+M+C", "n_CG_full_s7": "n_fake_CG_s7", "n_CG_full_s123": "n_fake_CG_s123",
                          "n_real": "n_real_orig"})
    C.to_csv(os.path.join(F.DATA, "fig2_candidates.csv"), index=False)
    with pd.option_context("display.width", 220):
        print(C[["version", "row", "patch_uid", "n_real_orig", "n_fake_CG", "n_fake_CG+M", "n_fake_CG+M+A",
                 "n_fake_CG+M+C", "eligible", "rule"]].to_string(index=False))
