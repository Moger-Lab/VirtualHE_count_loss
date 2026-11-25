"""
s03b -- the two v1 ablation models retrained on the v2 data, matched to A1.

  CG   plain CycleGAN: adversarial + cycle only. SRS and H&E patches are
       drawn INDEPENDENTLY (unpaired), no macro loss, no nuclear loss.
  CGM  CycleGAN + macro-L1 (weight 5, 8x8, warm-up 5), quasi-paired sampling.
       Identical to A1 minus the area nuclear loss.

Everything else is copied from s03_train_count.py / BASE_CONFIG so that CG,
CGM, A1 and A5 differ only in their defining loss terms: v2 arrays, per-core
SRS normalisation, ResNet 9 blocks, LSGAN with 0.9 smoothing, D lr 1e-4,
D loss x0.5, identity OFF, rot90, batch 2, 6076 samples per epoch, AMP,
30 epochs, the same 52 training cores (paired + unpaired-only; confidence 0
for unpaired-only and stain-defective cores masks the macro term).

s03_train_count.py is imported read-only (ARMS/BASE_CONFIG/data selection)
and is never modified.

Usage:
  python s03b_train_cg.py --arm CG  --epochs 30 --seed 42 --tag CG_full --samples_per_epoch 6076 --ckpt_every 1
  python s03b_train_cg.py --arm CGM --max_iters 20 --tag smoke_cgm        (smoke)
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
import torch

import countlib as cl
import countnet as cn
import s03_train_count as s03

ARMS_B = {
    "CG": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="none", DENSITY_SAMPLER=False,
               SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=8, LAMBDA_MACRO=0.0, INDEPENDENT_HE=True),
    "CGM": dict(SRS_NORM_MODE="per_core", NUCLEAR_LOSS="none", DENSITY_SAMPLER=False,
                SRS_CHANNEL_INDICES=[1], COUNT_GRID_SIZE=8, INDEPENDENT_HE=False),
}


class V2IndependentDataset(cn.V2PatchDataset):
    """Unpaired variant: index is a tuple (i, j); SRS comes from row i and H&E
    from row j (an independently drawn training row). confidence_weight is
    returned as 1.0 (no quasi-paired term is used with this dataset)."""

    def __getitem__(self, idx):
        i, j = idx
        a = super().__getitem__(i)
        b = super().__getitem__(j)
        return dict(srs=a["srs"], he=b["he"], confidence_weight=torch.tensor(1.0, dtype=torch.float32),
                    n_real=b["n_real"], patch_uid=a["patch_uid"], core=a["core"],
                    he_patch_uid=b["patch_uid"], he_core=b["core"])


class IndependentPairSampler:
    """Yields (i, j) tuples: i from a without-replacement random draw (same
    generator/stream as A1's RandomSampler, so the SRS order matches A1);
    j uniform over all training rows from a separate generator."""

    def __init__(self, n, num_samples, gen_i, gen_j):
        import torch
        self.n, self.num_samples, self.gen_i, self.gen_j = n, num_samples, gen_i, gen_j
        self._torch = torch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        t = self._torch
        i = t.randperm(self.n, generator=self.gen_i)[:self.num_samples].tolist()
        j = t.randint(0, self.n, (self.num_samples,), generator=self.gen_j).tolist()
        return iter(list(zip(i, j)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS_B))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--deterministic", type=int, default=0)
    ap.add_argument("--max_iters", type=int, default=0, help="smoke test: stop after N iterations")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_val_log", action="store_true")
    ap.add_argument("--samples_per_epoch", type=int, default=6076)
    ap.add_argument("--ckpt_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    CFG = dict(s03.BASE_CONFIG, **ARMS_B[args.arm])
    CFG["ARM"] = args.arm
    tag = args.tag or f"{args.arm}_full"
    out_dir = os.path.join(_script_dir, "outputs", "runs", tag)
    ck_dir = os.path.join(cl.WORK_DIR, "runs", tag)
    if os.path.exists(os.path.join(out_dir, "config.json")) and not args.resume:
        raise SystemExit(f"[{tag}] output folder already has a run; use a new --tag or --resume")
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
    CFG["SEED"] = args.seed
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, RandomSampler

    v1c, v1m = cn.v1_common, cn.v1_models

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

    train_rows, val_rows, cv = s03.build_rows("full", CFG["SEED"])
    train_rows.to_csv(os.path.join(out_dir, "train_rows.csv"), index=False)
    print(f"[{tag}] {len(train_rows)} train patches / {train_rows.core.nunique()} cores | "
          f"val {len(val_rows)} patches / {val_rows.core.nunique()} cores", flush=True)

    g_sampler = torch.Generator().manual_seed(CFG["SEED"])
    g_he = torch.Generator().manual_seed(CFG["SEED"] + 1)
    n_per_epoch = args.samples_per_epoch or len(train_rows)
    CFG["SAMPLES_PER_EPOCH"] = n_per_epoch
    if CFG["INDEPENDENT_HE"]:
        ds = V2IndependentDataset(train_rows, CFG, calib)
        sampler = IndependentPairSampler(len(ds), n_per_epoch, g_sampler, g_he)
    else:
        ds = cn.V2PatchDataset(train_rows, CFG, calib)
        sampler = RandomSampler(ds, replacement=False, num_samples=n_per_epoch, generator=g_sampler)
    loader = DataLoader(ds, batch_size=CFG["BATCH_SIZE"], sampler=sampler, num_workers=CFG["NUM_WORKERS"],
                        drop_last=True, pin_memory=True, persistent_workers=True,
                        worker_init_fn=v1c._dataloader_worker_init_fn)

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

    # val logging: paired val patches, dataset and metrics as in s03
    val_log_rows = s03.fixed_val_subset(val_rows, CFG["N_VAL_LOG"])
    val_log_ds = cn.V2PatchDataset(val_log_rows, CFG, calib)
    counter = cn.load_counter(os.path.join(cl.WORK_DIR, "counter", "counter_best.pt"), dev)

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

    latest = os.path.join(ck_dir, "latest.pt")
    start_epoch = 0
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev, weights_only=False)
        for k, net in (("G_S2H", G_S2H), ("G_H2S", G_H2S), ("D_H", D_H), ("D_S", D_S)):
            net.load_state_dict(ck[k])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        sch_G.load_state_dict(ck["sch_G"]); sch_D.load_state_dict(ck["sch_D"])
        torch.set_rng_state(ck["torch_rng"].cpu()); np.random.set_state(ck["np_rng"]); _random.setstate(ck["py_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"]); g_sampler.set_state(ck["sampler_rng"]); g_he.set_state(ck["he_rng"])
        start_epoch = ck["epoch"] + 1
        print(f"[{tag}] resumed at epoch {start_epoch}", flush=True)

    def save_full(epoch):
        torch.save(dict(epoch=epoch, G_S2H=G_S2H.state_dict(), G_H2S=G_H2S.state_dict(), D_H=D_H.state_dict(),
                        D_S=D_S.state_dict(), opt_G=opt_G.state_dict(), opt_D=opt_D.state_dict(),
                        sch_G=sch_G.state_dict(), sch_D=sch_D.state_dict(), torch_rng=torch.get_rng_state(),
                        np_rng=np.random.get_state(), py_rng=_random.getstate(),
                        cuda_rng=torch.cuda.get_rng_state_all(), sampler_rng=g_sampler.get_state(),
                        he_rng=g_he.get_state(), config=CFG), latest + ".tmp")
        os.replace(latest + ".tmp", latest)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(CFG, n_train=len(train_rows), train_cores=sorted(train_rows.core.unique()),
                       epochs=args.epochs, subset="full"), f, indent=1, default=str)

    log_path = os.path.join(out_dir, "loss_history.csv")
    if start_epoch == 0 and os.path.exists(log_path):
        os.remove(log_path)
    keys = ["loss_G", "loss_D", "loss_D_H", "loss_D_S", "loss_adv", "loss_cycle", "loss_macro", "d_real_H", "d_fake_H"]
    it_total = 0
    print(f"[{tag}] arm {args.arm}: norm {CFG['SRS_NORM_MODE']} | independent_he {CFG['INDEPENDENT_HE']} | "
          f"lambda_macro {CFG['LAMBDA_MACRO']} | channels {CFG['SRS_CHANNEL_INDICES']} | identity OFF | "
          f"det {CFG['DETERMINISTIC']} | AMP {amp} | {n_per_epoch} samples/epoch", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        for net in (G_S2H, G_H2S, D_H, D_S):
            net.train()
        macro_on = CFG["LAMBDA_MACRO"] > 0 and epoch >= CFG["WARMUP_EPOCHS"]
        acc = {k: [] for k in keys}
        for batch in loader:
            S = batch["srs"].to(dev, non_blocking=True); H = batch["he"].to(dev, non_blocking=True)
            w = batch["confidence_weight"].to(dev, non_blocking=True)
            if CFG["ROT90_AUG"]:
                S, H = v1c.random_rot90_pair(S, H)
            opt_G.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                fH = G_S2H(S); rS = G_H2S(fH); fS = G_H2S(H); rH = G_S2H(fS)
                pfH, pfS = D_H(fH), D_S(fS)
                l_adv = gan(pfH, tgt(pfH, True)) + gan(pfS, tgt(pfS, True))
                l_cyc = l1(rS, S) + l1(rH, H)
                loss_G = l_adv + CFG["LAMBDA_CYCLE"] * l_cyc
                if macro_on:
                    l_mac = macro_loss(fH, H, w)
                    loss_G = loss_G + CFG["LAMBDA_MACRO"] * l_mac
                else:
                    l_mac = torch.zeros((), device=dev)
            scaler_G.scale(loss_G).backward(); scaler_G.step(opt_G); scaler_G.update()

            opt_D.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                fHd, fSd = pool_H.query(fH.detach()), pool_S.query(fS.detach())
                prH, pfHd = D_H(H), D_H(fHd)
                lDH = 0.5 * (gan(prH, tgt(prH, True, True)) + gan(pfHd, tgt(pfHd, False)))
                prS, pfSd = D_S(S), D_S(fSd)
                lDS = 0.5 * (gan(prS, tgt(prS, True, True)) + gan(pfSd, tgt(pfSd, False)))
                loss_D = 0.5 * (lDH + lDS) if CFG["D_LOSS_HALF"] else (lDH + lDS)
            scaler_D.scale(loss_D).backward(); scaler_D.step(opt_D); scaler_D.update()

            vals = [loss_G, loss_D, lDH, lDS, l_adv, l_cyc, l_mac, prH.mean(), pfHd.mean()]
            for k, v in zip(keys, vals):
                acc[k].append(float(v.detach()))
            it_total += 1
            n_it = len(acc["loss_G"])
            if n_it % 250 == 0:
                print(f"[{tag}]   ep {epoch} iter {n_it}/{n_per_epoch // CFG['BATCH_SIZE']} "
                      f"{(time.time() - t0) / n_it:.3f}s/iter", flush=True)
            if args.max_iters and it_total >= args.max_iters:
                break
        sch_G.step(); sch_D.step()

        row = {"epoch": epoch, "time_s": round(time.time() - t0, 1), "iters": len(acc["loss_G"])}
        for k in keys:
            row[k] = float(np.nanmean(acc[k])); row[k + "_std"] = float(np.nanstd(acc[k]))
        if not np.isfinite(row["loss_G"]):
            print(f"[{tag}] NON-FINITE loss at epoch {epoch}; aborting", flush=True)
            sys.exit(2)
        if not args.max_iters:
            row.update(val_log(epoch))
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
        print(f"[{tag}] ep {epoch:3d} G {row['loss_G']:.3f} D {row['loss_D']:.3f} adv {row['loss_adv']:.3f} "
              f"cyc {row['loss_cycle']:.3f} mac {row['loss_macro']:.4f} "
              f"val sd fake/real {row.get('val_sd_fake_mean', np.nan):.1f}/{row.get('val_sd_real_mean', np.nan):.1f} "
              f"disc {row.get('val_disc_sd', np.nan):.3f} | {row['time_s']}s", flush=True)

        save_full(epoch)
        if ((epoch + 1) % CFG["CKPT_EVERY"] == 0 or epoch + 1 == args.epochs
                or epoch + 1 > args.epochs - CFG["LAST_K"]):
            torch.save(dict(epoch=epoch, G_S2H=G_S2H.state_dict(), config=CFG),
                       os.path.join(ck_dir, f"G_S2H_epoch{epoch + 1:03d}.pt"))
        if args.max_iters and it_total >= args.max_iters:
            print(f"[{tag}] smoke test stopped after {it_total} iterations: loss_G {row['loss_G']:.4f}", flush=True)
            break
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    main()
