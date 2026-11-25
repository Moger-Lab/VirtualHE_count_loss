"""
s18a -- metric/detector SCREEN, smoke-test inputs (srs-analysis env).

Picks a small density-stratified subset of the pooled held-out eval patches
(val fold 0 + legacy test), and for each patch writes one npz with:
  real_rgb       512x512 uint8, exactly as the StarDist cache saw it
  rt_rgb         he_to_rgb(rgb_to_he(real_rgb))  (R1 round trip)
  real_he        (2,512,512) float16, rgb_to_he of real_rgb, [-1, 1]
  fake_rgb_<M>   he_to_rgb of generator output, model M
  fake_he_<M>    (2,512,512) float16 generator output, [-1, 1]
for the four paper models at one seed and one epoch. Runs StarDist (v1
protocol) on real, round-trip and fake RGB and saves the label images, so
every detector is compared on identical pixels.

Read-only with respect to all existing results: writes only to
D:\\vhe_count_work\\s18_metric_screen\\<out> and outputs\\s18_metric_screen\\.

Usage: python s18a_smoke_inputs.py --n_low 7 --n_med 7 --n_high 6 --seed_tag "" --epoch 30
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

MODELS = {"CG": "CG_full", "CGM": "CGM_full", "CGMA": "A1_full", "CGMC": "A5_full"}
OUT_PROJ = os.path.join(_script_dir, "outputs", "s18_metric_screen")
OUT_WORK = os.path.join(cl.WORK_DIR, "s18_metric_screen")


def heldout_rows():
    m, _, _ = cl.load_v2_tables()
    m["split"] = cl.split_of(m, 0)
    rows = m[m.split.isin(["val", "legacy_test"]) & (m.role == "eval")].copy()
    sd = pd.read_csv(os.path.join(_script_dir, "outputs", "s02_counter", "stardist_per_patch.csv"))
    rows["n_real"] = rows.patch_uid.map(dict(zip(sd.patch_uid, sd.n_sd)))
    rows["confidence_weight"] = 1.0
    return rows.reset_index(drop=True)


def pick_stratified(rows, n_per_bucket, seed):
    """Within each density bucket: spread over count quantiles, at most one
    patch per core where possible."""
    rng = np.random.default_rng(seed)
    rows = rows.copy()
    rows["bucket"] = cl.density_bucket(rows.n_real.values)
    out = []
    for b, n in zip(cl.BUCKET_NAMES, n_per_bucket):
        d = rows[rows.bucket == b].sort_values("n_real").reset_index(drop=True)
        qs = np.linspace(0.05, 0.95, n)
        used = set()
        for q in qs:
            target = d.n_real.quantile(q)
            cand = d[~d.core.isin(used)] if (~d.core.isin(used)).any() else d
            cand = cand.iloc[(cand.n_real - target).abs().argsort()[:5]]
            r = cand.iloc[rng.integers(len(cand))]
            used.add(r.core); out.append(r)
            d = d[d.patch_uid != r.patch_uid]
    return pd.DataFrame(out).reset_index(drop=True)


def load_G(tag, epoch, dev):
    import torch
    import countnet as cn
    ck = torch.load(os.path.join(cl.WORK_DIR, "runs", tag, f"G_S2H_epoch{epoch:03d}.pt"), map_location=dev, weights_only=False)
    cfg = ck["config"]
    G = cn.v1_models.ResnetGenerator(len(cfg["SRS_CHANNEL_INDICES"]), 2, ngf=cfg["NGF"],
                                     n_residual_blocks=cfg["N_RESIDUAL_BLOCKS"],
                                     upsample_mode=cfg["UPSAMPLE_MODE"]).to(dev)
    G.load_state_dict(ck["G_S2H"]); G.eval()
    assert not cfg.get("INPUT_FLIP_LR", False)
    return G, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_low", type=int, default=7)
    ap.add_argument("--n_med", type=int, default=7)
    ap.add_argument("--n_high", type=int, default=6)
    ap.add_argument("--seed_tag", default="", help='"" (seed 42), "_s7" or "_s123"')
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--pick_seed", type=int, default=0)
    ap.add_argument("--out", default="smoke")
    a = ap.parse_args()

    import torch
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(12)
    import countnet as cn

    out_dir = os.path.join(OUT_WORK, a.out)
    if os.path.exists(os.path.join(out_dir, "patches.csv")):
        sys.exit(f"{out_dir} already populated; refusing to overwrite")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(OUT_PROJ, exist_ok=True)
    dev = torch.device("cuda")
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    sel = pick_stratified(heldout_rows(), (a.n_low, a.n_med, a.n_high), a.pick_seed)
    sc = cl.stardist_cache()
    gens = {k: load_G(v + a.seed_tag, a.epoch, dev) for k, v in MODELS.items()}

    timing = []
    recs = []
    for i, r in sel.iterrows():
        srs_n = None
        rec = dict(patch_uid=r.patch_uid, core=r.core, bucket=r.bucket, n_real_cache=int(r.n_real))
        store = {}
        _, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r["size"]))
        real_rgb = cl.resize_rgb(he, cl.RENDER)
        real_he = cn.v1_common.rgb_to_he(real_rgb, calib)
        rt_rgb = cn.v1_common.he_to_rgb(real_he[0], real_he[1], calib)
        store.update(real_rgb=real_rgb, rt_rgb=rt_rgb, real_he=real_he.astype(np.float16))
        for k, (G, cfg) in gens.items():
            ds = cn.V2PatchDataset(pd.DataFrame([r]), cfg, calib)
            s = ds[0]
            with torch.no_grad():
                fH = G(s["srs"][None].to(dev)).float()[0].cpu().numpy()
            store[f"fake_he_{k}"] = fH.astype(np.float16)
            store[f"fake_rgb_{k}"] = cn.v1_common.he_to_rgb(fH[0], fH[1], calib)
        for name in ["real_rgb", "rt_rgb"] + [f"fake_rgb_{k}" for k in MODELS]:
            t0 = time.perf_counter()
            (cy, cx, ar), lab = cl.detect_nuclei(store[name])
            timing.append(dict(detector="stardist", image=name, sec=time.perf_counter() - t0))
            store["sd_lab_" + name] = lab.astype(np.int32)
            rec["sd_n_" + name] = len(ar)
        ry, rx, ra = sc[r.patch_uid]
        rec["cache_match"] = int(len(ra) == rec["sd_n_real_rgb"])
        np.savez_compressed(os.path.join(out_dir, f"{r.patch_uid}.npz"), **store)
        recs.append(rec)
        print(f"{i + 1}/{len(sel)} {r.patch_uid} {r.bucket} real {rec['sd_n_real_rgb']} rt {rec['sd_n_rt_rgb']} "
              + " ".join(f"{k} {rec['sd_n_fake_rgb_' + k]}" for k in MODELS), flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out_dir, "patches.csv"), index=False)
    pd.DataFrame(timing).to_csv(os.path.join(out_dir, "timing_stardist.csv"), index=False)
    print(f"cache reproduced on {df.cache_match.sum()}/{len(df)} patches")
    print(f"stardist sec/patch median {pd.DataFrame(timing).sec.median():.2f}")
