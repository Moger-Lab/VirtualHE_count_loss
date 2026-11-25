"""
s02c -- train the U-Net nucleus counter (Step 2, part 3).

Input: 1-channel calibrated H in [-1, 1] (v1 rgb_to_he, exactly the
generator's output space). Target: StarDist heatmap, one unit-integral
Gaussian per centroid, sigma = 0.5 x median StarDist equivalent radius
(512-grid px, measured on the TRAIN split). Integral of target == count.

Train split only (v2 rotating folds 1-4, train tiling). Model selection on
the val split (fold 0, disjoint eval tiling) by count Pearson r. Legacy test
cores are never touched.

Loss: MSE on density x 1000 (the network's native scale) + COUNT_L1_WEIGHT x
|sum(pred) - N|.

Augmentation (so the counter does not key on generator texture): rot90 /
flips, Gaussian blur sigma 0-1.5 px, intensity scale +/-10% on stain amount,
additive noise with sd drawn in [0, 0.04].
"""

import json
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler

import countlib as cl
import countnet as cn

CFG = dict(
    SEED=7,
    EPOCHS=24,
    SAMPLES_PER_EPOCH=4000,
    BATCH_SIZE=8,
    LR=1e-3,
    WEIGHT_DECAY=1e-4,
    COUNT_L1_WEIGHT=0.005,
    NUM_WORKERS=6,
    BASE=32,
)
OUT_DIR = os.path.join(_script_dir, "outputs", "s02_counter")
CKPT_DIR = os.path.join(cl.WORK_DIR, "counter")


def load_rows():
    df = pd.read_csv(os.path.join(OUT_DIR, "stardist_per_patch.csv"))
    tr = df[(df.split == "train") & (df.role == "train")].reset_index(drop=True)
    va = df[(df.split == "val") & (df.role == "eval")].reset_index(drop=True)
    return tr, va


def evaluate(net, loader, device):
    net.eval()
    pred, true, uids = [], [], []
    with torch.no_grad():
        for b in loader:
            d = net(b["h"].to(device))
            pred.append(d.sum(dim=(1, 2, 3)).cpu().numpy()); true.append(b["count"].numpy())
            uids += list(b["patch_uid"])
    return np.concatenate(pred), np.concatenate(true), uids


def metrics(pred, true):
    r = float(np.corrcoef(pred, true)[0, 1])
    b = cl.density_bucket(true)
    out = dict(pearson_r=r, mae=float(np.mean(np.abs(pred - true))))
    for k in cl.BUCKET_NAMES:
        m = b == k
        out[f"ratio_{k}"] = float(np.median(pred[m] / np.maximum(true[m], 1))) if m.any() else np.nan
        out[f"n_{k}"] = int(m.sum())
    return out


if __name__ == "__main__":
    torch.manual_seed(CFG["SEED"]); np.random.seed(CFG["SEED"])
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda")
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    tr, va = load_rows()
    r_eq = cl.median_equiv_radius(tr.patch_uid)
    sigma = 0.5 * r_eq
    print(f"train {len(tr)} patches, val {len(va)}; median equiv radius {r_eq:.2f} px -> sigma {sigma:.2f} px")

    ds_tr = cn.CounterDataset(tr, calib, sigma, augment=True)
    ds_va = cn.CounterDataset(va, calib, sigma, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=CFG["BATCH_SIZE"], num_workers=CFG["NUM_WORKERS"], persistent_workers=True,
                       sampler=RandomSampler(ds_tr, replacement=True, num_samples=CFG["SAMPLES_PER_EPOCH"]),
                       worker_init_fn=cn.v1_common._dataloader_worker_init_fn, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=16, num_workers=CFG["NUM_WORKERS"], persistent_workers=True)

    net = cn.UNetCounter(CFG["BASE"]).to(device)
    print(f"counter params: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
    steps = CFG["EPOCHS"] * (CFG["SAMPLES_PER_EPOCH"] // CFG["BATCH_SIZE"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["LR"], total_steps=steps, pct_start=0.1)
    S = cn.UNetCounter.DENSITY_SCALE

    hist = []
    best_r = -1
    for ep in range(CFG["EPOCHS"]):
        net.train(); t0 = time.time(); tl = tm = tc = 0.0; n = 0
        for b in dl_tr:
            h = b["h"].to(device, non_blocking=True); hm = b["hm"].to(device, non_blocking=True)
            cnt = b["count"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps = net.forward_scaled(h)
            ps = ps.float()
            mse = torch.mean((ps - hm * S) ** 2)
            l1 = torch.mean(torch.abs(ps.sum(dim=(1, 2, 3)) / S - cnt))
            loss = mse + CFG["COUNT_L1_WEIGHT"] * l1
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); tm += mse.item(); tc += l1.item(); n += 1
        pred, true, _ = evaluate(net, dl_va, device)
        m = metrics(pred, true)
        rec = dict(epoch=ep, loss=tl / n, mse=tm / n, count_l1=tc / n, time_s=time.time() - t0, **m)
        hist.append(rec)
        print(f"ep {ep:2d} loss {rec['loss']:.4f} mse {rec['mse']:.4f} cntL1 {rec['count_l1']:.2f} | "
              f"val r {m['pearson_r']:.4f} mae {m['mae']:.2f} ratios low/med/high "
              f"{m['ratio_low']:.3f}/{m['ratio_med']:.3f}/{m['ratio_high']:.3f} | {rec['time_s']:.0f}s", flush=True)
        state = dict(state_dict=net.state_dict(), base=CFG["BASE"], sigma=sigma, epoch=ep, val=m, cfg=CFG)
        torch.save(state, os.path.join(CKPT_DIR, "counter_last.pt"))
        if m["pearson_r"] > best_r:
            best_r = m["pearson_r"]
            torch.save(state, os.path.join(CKPT_DIR, "counter_best.pt"))
        pd.DataFrame(hist).to_csv(os.path.join(OUT_DIR, "counter_training_history.csv"), index=False)
    with open(os.path.join(OUT_DIR, "counter_config.json"), "w") as f:
        json.dump(dict(CFG, sigma=sigma, median_equiv_radius=r_eq, best_val_r=best_r), f, indent=2)
    print(f"best val r {best_r:.4f}; checkpoint {os.path.join(CKPT_DIR, 'counter_best.pt')}")
