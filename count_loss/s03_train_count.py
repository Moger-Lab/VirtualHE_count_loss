"""
s03 -- Model 3 on v2 patches, with the count-based nuclear loss and the
other brief step-3 interventions, all flag-gated. One script for every arm.

Base: a copy of v1 Model 3's logic (05_train_morphology_nuclear_cyclegan.py,
RTX4500 folder). v1's files are imported read-only for the networks,
calibration and rgb_to_he / he_to_rgb, and are never modified.

Kept from Model 3 so arms stay comparable to it:
  LSGAN, one-sided label smoothing 0.9, image pools of 50, Adam(0.5, 0.999),
  G lr 2e-4, D lr 1e-4 (half rate), D loss = 0.5 * (0.5*(..H) + 0.5*(..S))
  (the 0.5x weighting of the reference Model 3, D_LOSS_HALF), cycle 10,
  macro-L1 5 @ 8x8 with 5 warmup epochs, per-sample rot90 augmentation
  (RTX4500 base), LR decay from epoch 75 of 100 (schedule fixed to the
  100-epoch plan even for 15-epoch screening, so screening = early phase).
Forced: identity loss OFF regardless of channel count (brief hard rule).

Arms (brief step 5):
  A0  Model 3 as-is on v2 patches (per-patch SRS norm, area nuclear loss)
  A1  A0 + per-core SRS normalisation
  A2  A1 + count loss (replaces area loss) + density sampler, grid 8
  A3  A2 + SRS/SHG/TPEF input (channels [1, 0, 3])
  A4  A2 with COUNT_GRID_SIZE = 2

Usage:
  python s03_train_count.py --arm A2 --subset screen --epochs 15
  python s03_train_count.py --arm A2 --subset screen --max_iters 2   (smoke)
  python s03_train_count.py --arm A2 --subset full --epochs 100 --resume
"""

import argparse
import json
import os
import random as _random
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd

import countlib as cl
import countnet as cn   # imports torch; CUDA is not initialised until first use

ARMS = {
    "A0": dict(SRS_NORM_MODE="per_patch", NUCLEAR_LOSS="area", DENSITY_SAMPLER=False, SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=8),
    "A1": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="area", DENSITY_SAMPLER=False, SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=8),
    "A2": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="count", DENSITY_SAMPLER=True, SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=8),
    "A3": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="count", DENSITY_SAMPLER=True, SRS_CHANNEL_INDICES=[1, 0, 3], COUNT_GRID_SIZE=8),
    "A4": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="count", DENSITY_SAMPLER=True, SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=2),
    # A5 (post-Step-6, D-15): A1 + LOCAL-ONLY count loss at 50 um cells; no sampler, 1 channel, so the
    # only difference from A1 is the per-cell count term replacing the area term
    "A5": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="count", DENSITY_SAMPLER=False, SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=2, COUNT_TOTAL_WEIGHT=0.0),
    # A7 (D-20): A5 + tissue-type channels (SRS, SHG, TPEF)
    "A7": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="count", DENSITY_SAMPLER=False, SRS_CHANNEL_INDICES=[1, 0, 3], COUNT_GRID_SIZE=2, COUNT_TOTAL_WEIGHT=0.0),
    # A6 (D-18): A5 + SRS-emptiness penalty (no stain where the SRS shows no tissue)
    "A6": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="count", DENSITY_SAMPLER=False, SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=2, COUNT_TOTAL_WEIGHT=0.0, SRS_EMPTY_LOSS=True),
}

BASE_CONFIG = dict(
    SEED=42,
    NETWORK_INPUT_SIZE=512,
    SRS_CORE_FLIP_HORIZONTAL=False,   # v2 path: the mirror is inside the warped H&E (D-01)
    HE_CORE_FLIP_HORIZONTAL=False,
    TYPE_CHANNEL_NORM="global",       # D-06
    BATCH_SIZE=2,
    NUM_WORKERS=4,
    NGF=64, NDF=64, N_RESIDUAL_BLOCKS=9, UPSAMPLE_MODE="nearest_conv", INIT_GAIN=0.02,
    LAMBDA_CYCLE=10.0,
    IDENTITY_LOSS=False,              # forced OFF (brief hard rule)
    LAMBDA_MACRO=5.0, MACRO_GRID_SIZE=8, WARMUP_EPOCHS=5,
    # Model 3's area-based nuclear loss (A0 / A1)
    LAMBDA_NUCLEAR=5.0, NUCLEAR_GRID_SIZE=8, NUCLEAR_SOFT_TEMPERATURE=0.1,
    NUCLEAR_OVER_WEIGHT=2.0, NUCLEAR_UNDER_WEIGHT=1.0, NUCLEAR_WARMUP_EPOCHS=5,
    # SRS-emptiness loss (A6, D-18): mean H+E stain amount over SRS-empty pixels
    SRS_EMPTY_LOSS=False, LAMBDA_EMPTY=2.0,
    # count loss (A2-A4)
    LAMBDA_COUNT=1.0,
    COUNT_TOTAL_WEIGHT=0.5,           # brief: L_cell + 0.5 * L_total
    LAMBDA_COUNT_MODE="auto",         # 'auto': rescale at end of epoch 4 so L_count ~ 20% of L_G
    LAMBDA_COUNT_TARGET_SHARE=0.20,
    COUNT_START_EPOCH=5, COUNT_FULL_EPOCH=10,
    DENSITY_SAMPLER_CAP=3.0,
    COUNTER_PATH=None,                # filled in main() from WORK_DIR
    # optimisation (Model 3)
    LEARNING_RATE=0.0002, LEARNING_RATE_D=0.0001, REAL_LABEL_SMOOTHING=0.9,
    ADAM_BETA1=0.5, ADAM_BETA2=0.999,
    SCHEDULE_EPOCHS=100, LR_DECAY_START_EPOCH=75,
    IMAGE_POOL_SIZE=50,
    D_LOSS_HALF=True,                 # reference Model 3 (non-aug ep69): loss_D = 0.5*(D_H + D_S)
    ROT90_AUG=True,                   # RTX4500 Model 3
    USE_MIXED_PRECISION=True,         # forced off by --deterministic
    CKPT_EVERY=5,
    LAST_K=4,
    N_VAL_LOG=36,                    # fixed val subset for per-epoch StarDist/counter logging
)

SCREEN_N_CORES = 8
SCREEN_MAX_PATCHES = 600
STAIN_DEFECTIVE = ("B-7", "C-7", "H-1")   # v1 FINDINGS 7 / v2 stage 4b


# =============================================================================
# DATA SELECTION
# =============================================================================

def screening_cores(cv: pd.DataFrame, val_fold: int = 0) -> list:
    """~8 training cores (folds != val_fold, rows B-H, stain-defective cores
    excluded), balanced across low / med / high core density and spread
    within each bucket. Deterministic."""
    c = cv[(cv.group == "rotating") & (cv.fold != val_fold) & (cv.row != "A")
           & (~cv.core.isin(STAIN_DEFECTIVE))].copy()
    c["bucket"] = cl.density_bucket(c.density_median.values)
    per = {"low": 3, "med": 2, "high": 3}
    chosen = []
    for b, k in per.items():
        s = c[c.bucket == b].sort_values("density_median")
        idx = np.unique(np.round(np.linspace(0, len(s) - 1, k)).astype(int))
        chosen += list(s.iloc[idx].core)
    return chosen


def build_rows(subset: str, seed: int):
    m, cv, tr = cl.load_v2_tables()
    m["split"] = cl.split_of(m, val_fold=0)
    sd = pd.read_csv(os.path.join(_script_dir, "outputs", "s02_counter", "stardist_per_patch.csv"))
    n_map = dict(zip(sd.patch_uid, sd.n_sd))
    m["n_real"] = m.patch_uid.map(n_map)
    m["confidence_weight"] = cn.confidence_weights(m)
    train = m[(m.split == "train") & (m.role == "train")]
    if subset == "screen":
        cores = screening_cores(cv)
        rng = np.random.default_rng(seed)
        per_core = SCREEN_MAX_PATCHES // len(cores)
        parts = []
        for c in cores:
            t = train[train.core == c]
            parts.append(t.iloc[np.sort(rng.choice(len(t), size=min(per_core, len(t)), replace=False))])
        train = pd.concat(parts)
    elif subset == "full":
        # unpaired-only cores join for adversarial + cycle; confidence 0 masks
        # the quasi-paired terms for them. Their n_real is unknown (not cached):
        # the density sampler treats them as their own stratum (see below).
        unp = m[(m.split == "unpaired") & (m.role == "train")]
        train = pd.concat([train, unp]).copy()
        # stain-defective cores: the counter reads the H channel, so their real
        # count target is biased low. Adversarial + cycle only (DECISION_LOG D-12).
        train.loc[train.core.isin(STAIN_DEFECTIVE), "confidence_weight"] = 0.0
    else:
        raise ValueError(subset)
    val = m[(m.split == "val") & (m.role == "eval")]
    return train.reset_index(drop=True), val.reset_index(drop=True), cv


def fixed_val_subset(val: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    b = cl.density_bucket(val.n_real.values)
    parts = []
    for k in cl.BUCKET_NAMES:
        idx = np.where(b == k)[0]
        parts.append(val.iloc[np.sort(rng.choice(idx, size=min(n // 3, len(idx)), replace=False))])
    return pd.concat(parts).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--subset", default="screen", choices=["screen", "full"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=0, help="smoke test: stop after N iterations")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lambda_count", type=float, default=None, help="fixed LAMBDA_COUNT (disables auto)")
    ap.add_argument("--no_val_log", action="store_true")
    ap.add_argument("--samples_per_epoch", type=int, default=0,
                    help="fix epoch length (0 = one pass over the training rows)")
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None, help="override SEED (replication runs)")
    args = ap.parse_args()

    CFG = dict(BASE_CONFIG, **ARMS[args.arm])
    CFG["ARM"] = args.arm
    CFG["COUNTER_PATH"] = os.path.join(cl.WORK_DIR, "counter", "counter_best.pt")
    if args.lambda_count is not None:
        CFG["LAMBDA_COUNT"], CFG["LAMBDA_COUNT_MODE"] = args.lambda_count, "fixed"
    tag = args.tag or f"{args.arm}_{args.subset}"
    out_dir = os.path.join(_script_dir, "outputs", "runs", tag)
    ck_dir = os.path.join(cl.WORK_DIR, "runs", tag)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(ck_dir, exist_ok=True)

    import torch
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        CFG["USE_MIXED_PRECISION"] = False
    CFG["DETERMINISTIC"] = bool(args.deterministic)
    CFG["CKPT_EVERY"] = args.ckpt_every
    if args.seed is not None:
        CFG["SEED"] = args.seed
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    v1c, v1m = cn.v1_common, cn.v1_models

    # --- models FIRST after seeding: identical init across same-channel arms
    v1c.set_seed(CFG["SEED"]); _random.seed(CFG["SEED"])
    dev = torch.device("cuda")
    n_srs = len(CFG["SRS_CHANNEL_INDICES"]); n_he = 2
    G_S2H = v1m.ResnetGenerator(n_srs, n_he, ngf=CFG["NGF"], n_residual_blocks=CFG["N_RESIDUAL_BLOCKS"], upsample_mode=CFG["UPSAMPLE_MODE"]).to(dev)
    G_H2S = v1m.ResnetGenerator(n_he, n_srs, ngf=CFG["NGF"], n_residual_blocks=CFG["N_RESIDUAL_BLOCKS"], upsample_mode=CFG["UPSAMPLE_MODE"]).to(dev)
    D_H = v1m.PatchGANDiscriminator(n_he, ndf=CFG["NDF"]).to(dev)
    D_S = v1m.PatchGANDiscriminator(n_srs, ndf=CFG["NDF"]).to(dev)
    for net in (G_S2H, G_H2S, D_H, D_S):
        net.apply(lambda mod: v1m.init_weights(mod, CFG["INIT_GAIN"]))

    calib = v1c.load_stain_calibration(cl.CALIBRATION_PATH)
    thr = float(calib.nuclear_h_threshold)

    train_rows, val_rows, cv = build_rows(args.subset, CFG["SEED"])
    train_rows.to_csv(os.path.join(out_dir, "train_rows.csv"), index=False)
    print(f"[{tag}] {len(train_rows)} train patches / {train_rows.core.nunique()} cores "
          f"{sorted(train_rows.core.unique()) if args.subset == 'screen' else ''} | val {len(val_rows)} patches / "
          f"{val_rows.core.nunique()} cores", flush=True)

    counter = cn.load_counter(CFG["COUNTER_PATH"], dev)
    paired_uids = train_rows[train_rows.confidence_weight > 0].patch_uid
    q95 = cl.q95_cell_counts(paired_uids, CFG["COUNT_GRID_SIZE"])
    CFG["Q95"] = q95
    print(f"[{tag}] counter {CFG['COUNTER_PATH']} (frozen) | COUNT_GRID_SIZE {CFG['COUNT_GRID_SIZE']} "
          f"({100.0 / CFG['COUNT_GRID_SIZE']:.1f} um cells) | Q95 per-cell real count {q95:.2f}", flush=True)

    ds = cn.V2PatchDataset(train_rows, CFG, calib)
    g_sampler = torch.Generator().manual_seed(CFG["SEED"])
    n_per_epoch = args.samples_per_epoch or len(ds)
    CFG["SAMPLES_PER_EPOCH"] = n_per_epoch
    if CFG["DENSITY_SAMPLER"]:
        n_real = train_rows.n_real.values.astype(float)
        if np.isnan(n_real).any():
            # unpaired cores (full run only): place them in the stratum of
            # the median so they keep roughly uniform weight
            n_real = np.where(np.isnan(n_real), np.nanmedian(n_real), n_real)
        sampler, p = cn.build_density_sampler(n_real, num_samples=n_per_epoch, cap=CFG["DENSITY_SAMPLER_CAP"], generator=g_sampler)
        b = cl.density_bucket(n_real)
        print(f"[{tag}] density sampler: natural " + ", ".join(f"{k} {np.mean(b == k):.2f}" for k in cl.BUCKET_NAMES)
              + " -> target " + ", ".join(f"{k} {p[b == k].sum():.2f}" for k in cl.BUCKET_NAMES)
              + f" | max oversampling {p.max() * len(p):.2f}x", flush=True)
        loader = DataLoader(ds, batch_size=CFG["BATCH_SIZE"], sampler=sampler, num_workers=CFG["NUM_WORKERS"],
                            drop_last=True, pin_memory=True, persistent_workers=True,
                            worker_init_fn=v1c._dataloader_worker_init_fn)
    else:
        from torch.utils.data import RandomSampler
        loader = DataLoader(ds, batch_size=CFG["BATCH_SIZE"],
                            sampler=RandomSampler(ds, replacement=False, num_samples=n_per_epoch, generator=g_sampler),
                            num_workers=CFG["NUM_WORKERS"], drop_last=True, pin_memory=True,
                            persistent_workers=True, worker_init_fn=v1c._dataloader_worker_init_fn)

    opt_G = torch.optim.Adam(list(G_S2H.parameters()) + list(G_H2S.parameters()), lr=CFG["LEARNING_RATE"],
                             betas=(CFG["ADAM_BETA1"], CFG["ADAM_BETA2"]))
    opt_D = torch.optim.Adam(list(D_H.parameters()) + list(D_S.parameters()), lr=CFG["LEARNING_RATE_D"],
                             betas=(CFG["ADAM_BETA1"], CFG["ADAM_BETA2"]))

    def lr_lambda(ep):
        s, tot = CFG["LR_DECAY_START_EPOCH"], CFG["SCHEDULE_EPOCHS"]
        return 1.0 if ep < s else max(0.0, 1.0 - (ep - s) / max(tot - s, 1))
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    gan, l1 = nn.MSELoss(), nn.L1Loss()
    pool_H, pool_S = v1m.ImagePool(CFG["IMAGE_POOL_SIZE"]), v1m.ImagePool(CFG["IMAGE_POOL_SIZE"])
    amp = bool(CFG["USE_MIXED_PRECISION"])
    scaler_G = torch.amp.GradScaler("cuda", enabled=amp); scaler_D = torch.amp.GradScaler("cuda", enabled=amp)

    def tgt(p, real, smooth=False):
        return torch.full_like(p, CFG["REAL_LABEL_SMOOTHING"] if smooth else 1.0) if real else torch.zeros_like(p)

    def macro_loss(f, r, w):
        g = CFG["MACRO_GRID_SIZE"]
        return (torch.abs(F.adaptive_avg_pool2d(f, (g, g)) - F.adaptive_avg_pool2d(r, (g, g))).mean(dim=[1, 2, 3]) * w).mean()

    def area_nuclear_loss(f, r, w):
        """Model 3's loss, unchanged: soft-thresholded H-positive AREA
        fraction on an 8x8 grid, 2:1 over:under."""
        def dens(h):
            return F.adaptive_avg_pool2d(torch.sigmoid((h - thr) / CFG["NUCLEAR_SOFT_TEMPERATURE"]), (CFG["NUCLEAR_GRID_SIZE"],) * 2)
        d = dens(f[:, 0:1]) - dens(r[:, 0:1])
        ps = (torch.relu(d) * CFG["NUCLEAR_OVER_WEIGHT"] + torch.relu(-d) * CFG["NUCLEAR_UNDER_WEIGHT"]).mean(dim=[1, 2, 3])
        return (ps * w).mean()

    # --- per-epoch validation logging on a fixed val subset
    val_log_rows = fixed_val_subset(val_rows, CFG["N_VAL_LOG"])
    val_log_ds = cn.V2PatchDataset(val_log_rows, CFG, calib)

    def val_log(epoch):
        if args.no_val_log:
            return {}
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(8)
        except RuntimeError:
            pass
        G_S2H.eval()
        sd_f, ct_f, ct_r, n_r = [], [], [], []
        with torch.no_grad():
            for i in range(len(val_log_ds)):
                s = val_log_ds[i]
                fH = G_S2H(s["srs"][None].to(dev)).float()
                ct_f.append(counter(fH[:, 0:1]).sum().item())
                ct_r.append(counter(s["he"][None, 0:1].to(dev)).sum().item())
                rgb = v1c.he_to_rgb(fH[0, 0].cpu().numpy(), fH[0, 1].cpu().numpy(), calib)
                (_, _, a), _ = cl.detect_nuclei(rgb)
                sd_f.append(len(a)); n_r.append(float(s["n_real"]))
        G_S2H.train()
        sd_f, ct_f, ct_r, n_r = map(np.asarray, (sd_f, ct_f, ct_r, n_r))
        b = cl.density_bucket(n_r)
        rec = dict(val_sd_fake_mean=sd_f.mean(), val_counter_fake_mean=ct_f.mean(), val_sd_real_mean=n_r.mean(),
                   val_counter_real_mean=ct_r.mean(),
                   val_counter_gap_rel=(ct_f.mean() - sd_f.mean()) / max(sd_f.mean(), 1e-6))
        for k in cl.BUCKET_NAMES:
            mk = b == k
            rec[f"val_ratio_sd_{k}"] = float(np.median(sd_f[mk] / np.maximum(n_r[mk], 1)))
        rec["val_disc_sd"] = rec["val_ratio_sd_high"] / max(rec["val_ratio_sd_low"], 1e-6)
        return rec

    # --- checkpointing / resume
    latest = os.path.join(ck_dir, "latest.pt")
    lam_count_full = CFG["LAMBDA_COUNT"]
    start_epoch = 0
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev, weights_only=False)
        for k, net in (("G_S2H", G_S2H), ("G_H2S", G_H2S), ("D_H", D_H), ("D_S", D_S)):
            net.load_state_dict(ck[k])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        sch_G.load_state_dict(ck["sch_G"]); sch_D.load_state_dict(ck["sch_D"])
        torch.set_rng_state(ck["torch_rng"].cpu()); np.random.set_state(ck["np_rng"]); _random.setstate(ck["py_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"]); g_sampler.set_state(ck["sampler_rng"])
        lam_count_full = ck["lambda_count_full"]
        start_epoch = ck["epoch"] + 1
        print(f"[{tag}] resumed at epoch {start_epoch} (lambda_count_full {lam_count_full})", flush=True)

    def save_full(epoch):
        torch.save(dict(epoch=epoch, G_S2H=G_S2H.state_dict(), G_H2S=G_H2S.state_dict(), D_H=D_H.state_dict(),
                        D_S=D_S.state_dict(), opt_G=opt_G.state_dict(), opt_D=opt_D.state_dict(),
                        sch_G=sch_G.state_dict(), sch_D=sch_D.state_dict(), torch_rng=torch.get_rng_state(),
                        np_rng=np.random.get_state(), py_rng=_random.getstate(),
                        cuda_rng=torch.cuda.get_rng_state_all(), sampler_rng=g_sampler.get_state(),
                        lambda_count_full=lam_count_full, config=CFG), latest + ".tmp")
        os.replace(latest + ".tmp", latest)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(CFG, n_train=len(train_rows), train_cores=sorted(train_rows.core.unique()),
                       epochs=args.epochs, subset=args.subset), f, indent=1, default=str)

    log_path = os.path.join(out_dir, "loss_history.csv")
    if start_epoch == 0 and os.path.exists(log_path):
        os.remove(log_path)
    keys = ["loss_G", "loss_D", "loss_D_H", "loss_D_S", "loss_adv", "loss_cycle", "loss_macro",
            "loss_nuclear_area", "loss_count", "loss_count_monitor", "count_share", "d_real_H", "d_fake_H",
            "fake_N_counter", "real_N_counter", "loss_empty"]
    it_total = 0
    use_count = CFG["NUCLEAR_LOSS"] == "count"
    print(f"[{tag}] arm {args.arm}: norm {CFG['SRS_NORM_MODE']} | nuclear {CFG['NUCLEAR_LOSS']} | sampler "
          f"{CFG['DENSITY_SAMPLER']} | channels {CFG['SRS_CHANNEL_INDICES']} | identity OFF | det {CFG['DETERMINISTIC']} "
          f"| AMP {amp}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        for net in (G_S2H, G_H2S, D_H, D_S):
            net.train()
        macro_on = epoch >= CFG["WARMUP_EPOCHS"]
        area_on = (CFG["NUCLEAR_LOSS"] == "area") and epoch >= CFG["NUCLEAR_WARMUP_EPOCHS"]
        lam_c = cn.lambda_count_schedule(epoch, lam_count_full, CFG["COUNT_START_EPOCH"], CFG["COUNT_FULL_EPOCH"]) if use_count else 0.0
        acc = {k: [] for k in keys}
        for batch in loader:
            S = batch["srs"].to(dev, non_blocking=True); H = batch["he"].to(dev, non_blocking=True)
            w = batch["confidence_weight"].to(dev, non_blocking=True)
            if CFG["ROT90_AUG"]:
                if CFG["SRS_EMPTY_LOSS"]:
                    # same RNG calls as v1's random_rot90_pair, also rotating the empty mask
                    E = batch["srs_empty"].to(dev, non_blocking=True)
                    S, H, E = S.clone(), H.clone(), E.clone()
                    for i in range(S.shape[0]):
                        k = torch.randint(0, 4, (1,)).item()
                        if k > 0:
                            S[i] = torch.rot90(S[i], k, dims=[-2, -1]); H[i] = torch.rot90(H[i], k, dims=[-2, -1])
                            E[i] = torch.rot90(E[i], k, dims=[-2, -1])
                else:
                    S, H = v1c.random_rot90_pair(S, H)
            opt_G.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                fH = G_S2H(S); rS = G_H2S(fH); fS = G_H2S(H); rH = G_S2H(fS)
                pfH, pfS = D_H(fH), D_S(fS)
                l_adv = gan(pfH, tgt(pfH, True)) + gan(pfS, tgt(pfS, True))
                l_cyc = l1(rS, S) + l1(rH, H)          # all SRS channels
                loss_G = l_adv + CFG["LAMBDA_CYCLE"] * l_cyc
                l_mac = macro_loss(fH, H, w) if macro_on else torch.zeros((), device=dev)
                loss_G = loss_G + (CFG["LAMBDA_MACRO"] * l_mac if macro_on else 0.0)
                if CFG["SRS_EMPTY_LOSS"]:
                    # stain amount in [0, 2]: 0 = white (both channels at -1)
                    stain = (fH[:, 0:1].float() + 1.0) * 0.5 + (fH[:, 1:2].float() + 1.0) * 0.5
                    l_empty = (stain * E).sum() / (E.sum() + 1.0)
                    if macro_on:
                        loss_G = loss_G + CFG["LAMBDA_EMPTY"] * l_empty
                else:
                    l_empty = torch.zeros((), device=dev)
                l_area = area_nuclear_loss(fH, H, w) if area_on else torch.zeros((), device=dev)
                if area_on:
                    loss_G = loss_G + CFG["LAMBDA_NUCLEAR"] * l_area
            # count loss: fp32, outside autocast, frozen counter
            if lam_c > 0:
                l_cnt, parts = cn.count_loss(counter, fH, H, w, CFG["COUNT_GRID_SIZE"], q95, return_parts=True, total_weight=CFG["COUNT_TOTAL_WEIGHT"])
                loss_G_total = loss_G.float() + lam_c * l_cnt
                l_cnt_mon = l_cnt.detach()
            else:
                with torch.no_grad():
                    l_cnt_mon, parts = cn.count_loss(counter, fH.detach(), H, w, CFG["COUNT_GRID_SIZE"], q95, return_parts=True, total_weight=CFG["COUNT_TOTAL_WEIGHT"])
                l_cnt = torch.zeros((), device=dev)
                loss_G_total = loss_G
            scaler_G.scale(loss_G_total).backward(); scaler_G.step(opt_G); scaler_G.update()

            opt_D.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                fHd, fSd = pool_H.query(fH.detach()), pool_S.query(fS.detach())
                prH, pfHd = D_H(H), D_H(fHd)
                lDH = 0.5 * (gan(prH, tgt(prH, True, True)) + gan(pfHd, tgt(pfHd, False)))
                prS, pfSd = D_S(S), D_S(fSd)
                lDS = 0.5 * (gan(prS, tgt(prS, True, True)) + gan(pfSd, tgt(pfSd, False)))
                loss_D = 0.5 * (lDH + lDS) if CFG["D_LOSS_HALF"] else (lDH + lDS)
            scaler_D.scale(loss_D).backward(); scaler_D.step(opt_D); scaler_D.update()

            share = float(lam_c * l_cnt.item() / max(loss_G_total.item(), 1e-9)) if lam_c > 0 else 0.0
            vals = [loss_G_total, loss_D, lDH, lDS, l_adv, l_cyc, l_mac, l_area, l_cnt, l_cnt_mon, share,
                    prH.mean(), pfHd.mean(), parts["N_f"].mean(), parts["N_r"].mean(), l_empty]
            for k, v in zip(keys, vals):
                acc[k].append(float(v.detach()) if torch.is_tensor(v) else float(v))
            it_total += 1
            n_it = len(acc["loss_G"])
            if n_it % 250 == 0:
                print(f"[{tag}]   ep {epoch} iter {n_it}/{n_per_epoch // CFG['BATCH_SIZE']} "
                      f"{(time.time() - t0) / n_it:.3f}s/iter", flush=True)
            if args.max_iters and it_total >= args.max_iters:
                break
        sch_G.step(); sch_D.step()

        row = {"epoch": epoch, "time_s": round(time.time() - t0, 1), "lambda_count": lam_c, "iters": len(acc["loss_G"])}
        for k in keys:
            row[k] = float(np.nanmean(acc[k])); row[k + "_std"] = float(np.nanstd(acc[k]))
        if not np.isfinite(row["loss_G"]):
            print(f"[{tag}] NON-FINITE loss at epoch {epoch}; aborting", flush=True)
            sys.exit(2)

        # auto-rescale LAMBDA_COUNT at the end of the last warmup epoch:
        # lambda * L_count = share/(1-share) * L_G(other terms)
        if use_count and CFG["LAMBDA_COUNT_MODE"] == "auto" and epoch == CFG["COUNT_START_EPOCH"] - 1:
            s = CFG["LAMBDA_COUNT_TARGET_SHARE"]
            # macro turns on at epoch 5 too: add its expected contribution
            # using the current macro magnitude (computed on the fly here)
            lg = row["loss_G"]
            with torch.no_grad():
                mac = []
                for bi, b2 in enumerate(loader):
                    if bi >= 20:
                        break
                    S2 = b2["srs"].to(dev); H2 = b2["he"].to(dev); w2 = b2["confidence_weight"].to(dev)
                    mac.append(macro_loss(G_S2H(S2).float(), H2, w2).item())
            lg_expected = lg + CFG["LAMBDA_MACRO"] * float(np.mean(mac))
            lam_count_full = s / (1 - s) * lg_expected / max(row["loss_count_monitor"], 1e-9)
            print(f"[{tag}] LAMBDA_COUNT auto-rescaled: L_G(other, incl. macro) ~ {lg_expected:.3f}, "
                  f"L_count ~ {row['loss_count_monitor']:.4f} -> LAMBDA_COUNT = {lam_count_full:.4f} "
                  f"(target {100 * s:.0f}% of L_G at full ramp)", flush=True)
        row["lambda_count_full"] = lam_count_full if use_count else 0.0

        if not args.max_iters:
            row.update(val_log(epoch))
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
        vl = (f" | val sd fake/real {row.get('val_sd_fake_mean', np.nan):.1f}/{row.get('val_sd_real_mean', np.nan):.1f}"
              f" counter-fake {row.get('val_counter_fake_mean', np.nan):.1f} gap {row.get('val_counter_gap_rel', np.nan):+.3f}"
              f" disc {row.get('val_disc_sd', np.nan):.3f}")
        print(f"[{tag}] ep {epoch:3d} G {row['loss_G']:.3f} D {row['loss_D']:.3f} adv {row['loss_adv']:.3f} "
              f"cyc {row['loss_cycle']:.3f} mac {row['loss_macro']:.4f} area {row['loss_nuclear_area']:.4f} "
              f"cnt {row['loss_count_monitor']:.3f} (lam {lam_c:.3f}, share {row['count_share']:.2f}) "
              f"N f/r {row['fake_N_counter']:.1f}/{row['real_N_counter']:.1f}{vl} | {row['time_s']}s", flush=True)
        if use_count and epoch == CFG["COUNT_FULL_EPOCH"]:
            sh = row["count_share"]
            print(f"[{tag}] epoch {epoch}: L_count share of L_G = {100 * sh:.1f}% "
                  f"{'(within 10-30%)' if 0.10 <= sh <= 0.30 else '(OUTSIDE 10-30% target)'}", flush=True)

        save_full(epoch)
        # every CKPT_EVERY epochs, plus the last LAST_K epochs so arms can be
        # judged on a multi-checkpoint mean (v1: lag-1 autocorrelation is
        # negative, a 4-checkpoint mean cut disc noise 0.023 -> 0.004)
        if ((epoch + 1) % CFG["CKPT_EVERY"] == 0 or epoch + 1 == args.epochs
                or epoch + 1 > args.epochs - CFG["LAST_K"]):
            torch.save(dict(epoch=epoch, G_S2H=G_S2H.state_dict(), config=CFG, lambda_count_full=lam_count_full),
                       os.path.join(ck_dir, f"G_S2H_epoch{epoch + 1:03d}.pt"))
        if args.max_iters and it_total >= args.max_iters:
            print(f"[{tag}] smoke test stopped after {it_total} iterations: loss_G {row['loss_G']:.4f}", flush=True)
            break
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    main()
