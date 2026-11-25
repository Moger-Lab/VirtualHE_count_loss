"""
s05 -- evaluate screening arms on VALIDATION (v2 fold 0, 496 disjoint eval
patches, 11 cores). Never touches legacy_test.

Per checkpoint (cached per file):  StarDist on fake RGB (v1 protocol), the
frozen counter on fake H, per-cell (8x8, 12.5 um) fake and real StarDist
counts.

Per arm, reported on the MEAN OF THE LAST 4 CHECKPOINTS (v1: single
checkpoints swing; lag-1 autocorrelation negative) and per checkpoint:
  per-bucket median fake/real ratio + core-cluster bootstrap 95% CI
  disc = high-bucket ratio / low-bucket ratio   (primary, FINDINGS 6)
  log-log slope of fake on real count
  global bias (median ratio, ratio of means)
  stromal false positives: fake nuclei per 12.5 um cell where the real
    count is 0, and fraction of such cells with >= 1 fake nucleus
  counter-vs-StarDist gap on fakes (relative)
  loss_D / loss_adv curves (from the training log)

Usage:  python s05_eval_arms.py --tags A0_screen A1_screen ... [--last_k 4]
"""

import argparse
import glob
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd

import countlib as cl

OUT_DIR = os.path.join(_script_dir, "outputs", "s05_screening")
GRID = 8


def eval_checkpoint(path, val_rows, calib, dev, counter, split="val"):
    import torch
    import countnet as cn
    cache = os.path.join(os.path.dirname(path), os.path.basename(path).replace(".pt", f"_{split}_eval.csv"))
    if os.path.exists(cache):
        return pd.read_csv(cache)
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck["config"]
    n_srs = len(cfg["SRS_CHANNEL_INDICES"])
    G = cn.v1_models.ResnetGenerator(n_srs, 2, ngf=cfg["NGF"], n_residual_blocks=cfg["N_RESIDUAL_BLOCKS"],
                                     upsample_mode=cfg["UPSAMPLE_MODE"]).to(dev)
    G.load_state_dict(ck["G_S2H"]); G.eval()
    ds = cn.V2PatchDataset(val_rows, cfg, calib)
    sc = cl.stardist_cache()
    recs = []
    for i in range(len(ds)):
        s = ds[i]
        with torch.no_grad():
            if cfg.get("INPUT_FLIP_LR", False):
                # Model 3 was trained on horizontally flipped SRS: present the
                # input that way and map the output back to the v2 frame (D-17)
                fH = torch.flip(G(torch.flip(s["srs"][None].to(dev), dims=[-1])).float(), dims=[-1])
            else:
                fH = G(s["srs"][None].to(dev)).float()
            n_ct = counter(fH[:, 0:1]).sum().item()
        rgb = cn.v1_common.he_to_rgb(fH[0, 0].cpu().numpy(), fH[0, 1].cpu().numpy(), calib)
        (fy, fx, fa), _ = cl.detect_nuclei(rgb)
        ry, rx, ra = sc[s["patch_uid"]]
        cf = cl.cell_counts_from_centroids(fy, fx, GRID); cr = cl.cell_counts_from_centroids(ry, rx, GRID)
        empty = cr == 0
        recs.append(dict(patch_uid=s["patch_uid"], core=s["core"], n_real=len(ra), n_fake=len(fa), n_fake_counter=n_ct,
                         n_empty_cells=int(empty.sum()), fake_in_empty=float(cf[empty].sum()),
                         empty_cells_with_fake=int((cf[empty] > 0).sum()),
                         med_area_fake=float(np.median(fa)) if len(fa) else np.nan))
    df = pd.DataFrame(recs)
    df.to_csv(cache, index=False)
    return df


def metrics(df):
    b = cl.density_bucket(df.n_real.values)
    ratio = df.n_fake / df.n_real.clip(lower=1)
    out = {}
    for k in cl.BUCKET_NAMES:
        out[f"ratio_{k}"] = float(np.median(ratio[b == k]))
    out["disc"] = out["ratio_high"] / max(out["ratio_low"], 1e-9)
    out["ratio_all_median"] = float(np.median(ratio))
    out["ratio_of_means"] = float(df.n_fake.mean() / df.n_real.mean())
    m = (df.n_real > 0) & (df.n_fake > 0)
    out["loglog_slope"] = float(np.polyfit(np.log(df.n_real[m]), np.log(df.n_fake[m]), 1)[0])
    out["stromal_fp_per_empty_cell"] = float(df.fake_in_empty.sum() / max(df.n_empty_cells.sum(), 1))
    out["stromal_fp_cell_frac"] = float(df.empty_cells_with_fake.sum() / max(df.n_empty_cells.sum(), 1))
    out["counter_gap_rel"] = float((df.n_fake_counter.mean() - df.n_fake.mean()) / max(df.n_fake.mean(), 1e-9))
    out["pearson_r"] = float(np.corrcoef(df.n_real, df.n_fake)[0, 1])
    return out


def cluster_bootstrap(df, n=1000, seed=0):
    """Resample cores with replacement (patches within a core are not
    independent), recompute the metrics."""
    rng = np.random.default_rng(seed)
    cores = df.core.unique()
    groups = {c: df[df.core == c] for c in cores}
    bs = []
    for _ in range(n):
        pick = rng.choice(cores, len(cores))
        d = pd.concat([groups[c] for c in pick])
        try:
            bs.append(metrics(d))
        except Exception:
            continue
    return pd.DataFrame(bs)


def mean_over_ckpts(dfs):
    """Average per-patch fake quantities across checkpoints (same patches)."""
    base = dfs[0][["patch_uid", "core", "n_real", "n_empty_cells"]].copy()
    for col in ("n_fake", "n_fake_counter", "fake_in_empty", "empty_cells_with_fake"):
        base[col] = np.mean([d[col].values for d in dfs], axis=0)
    return base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--last_k", type=int, default=4)
    ap.add_argument("--eval_only", action="store_true", help="evaluate checkpoints, skip the report")
    ap.add_argument("--epochs", type=int, nargs="*", default=None,
                    help="only these checkpoint epochs (1-indexed); the report averages the last --last_k of them")
    ap.add_argument("--split", default="val", choices=["val", "legacy_test"],
                    help="legacy_test is for the final Model 3 comparison only, never for selection")
    ap.add_argument("--out_suffix", default="")
    a = ap.parse_args()
    OUT_DIR = OUT_DIR + a.out_suffix
    os.makedirs(OUT_DIR, exist_ok=True)

    import torch
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(12)
    import countnet as cn
    dev = torch.device("cuda")
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    counter = cn.load_counter(os.path.join(cl.WORK_DIR, "counter", "counter_best.pt"), dev)
    m, _, _ = cl.load_v2_tables()
    m["split"] = cl.split_of(m, 0)
    val = m[(m.split == a.split) & (m.role == "eval")].copy()
    val["confidence_weight"] = 1.0
    sd = pd.read_csv(os.path.join(_script_dir, "outputs", "s02_counter", "stardist_per_patch.csv"))
    val["n_real"] = val.patch_uid.map(dict(zip(sd.patch_uid, sd.n_sd)))
    val = val.reset_index(drop=True)

    per_ckpt_rows, arm_rows, per_arm_patch = [], [], {}
    for tag in a.tags:
        ck_paths = sorted(glob.glob(os.path.join(cl.WORK_DIR, "runs", tag, "G_S2H_epoch*.pt")))
        if not ck_paths:
            print(f"{tag}: no checkpoints"); continue
        dfs = {}
        for p in ck_paths:
            ep = int(os.path.basename(p)[len("G_S2H_epoch"):-3])
            if a.epochs is not None and ep not in a.epochs:
                continue
            print(f"{tag} epoch {ep}: evaluating {len(val)} {a.split} patches...", flush=True)
            dfs[ep] = eval_checkpoint(p, val, calib, dev, counter, a.split)
            per_ckpt_rows.append(dict(arm=tag, epoch=ep, **metrics(dfs[ep])))
        if a.eval_only:
            continue
        last = sorted(dfs)[-a.last_k:]
        mdf = mean_over_ckpts([dfs[e] for e in last])
        per_arm_patch[tag] = mdf
        mm = metrics(mdf)
        bs = cluster_bootstrap(mdf)
        rec = dict(arm=tag, checkpoints=",".join(map(str, last)), **mm)
        for k in ("ratio_low", "ratio_med", "ratio_high", "disc", "ratio_all_median", "stromal_fp_per_empty_cell", "loglog_slope"):
            rec[f"{k}_ci_lo"], rec[f"{k}_ci_hi"] = np.percentile(bs[k], [2.5, 97.5])
        hist = pd.read_csv(os.path.join(_script_dir, "outputs", "runs", tag, "loss_history.csv"))
        rec["loss_D_final"] = hist.loss_D.iloc[-1]; rec["loss_adv_final"] = hist.loss_adv.iloc[-1]
        arm_rows.append(rec)

    pd.DataFrame(per_ckpt_rows).to_csv(os.path.join(OUT_DIR, "per_checkpoint_metrics.csv"), index=False)
    if a.eval_only or not arm_rows:
        sys.exit(0)
    arms = pd.DataFrame(arm_rows)
    arms.to_csv(os.path.join(OUT_DIR, "arm_summary.csv"), index=False)
    for k, name in (("disc", "rank_disc"), ("ratio_high", "rank_high_bias")):
        arms[name] = arms[k].rank(ascending=False).astype(int)
    arms["rank_global_bias"] = (arms.ratio_all_median - 1).abs().rank().astype(int)
    arms["rank_stromal_fp"] = arms.stromal_fp_per_empty_cell.rank().astype(int)
    cols = ["arm", "ratio_low", "ratio_med", "ratio_high", "disc", "disc_ci_lo", "disc_ci_hi", "loglog_slope",
            "ratio_all_median", "stromal_fp_per_empty_cell", "counter_gap_rel", "pearson_r",
            "rank_high_bias", "rank_disc", "rank_global_bias", "rank_stromal_fp"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(arms[cols].to_string(index=False))
    arms.to_csv(os.path.join(OUT_DIR, "arm_summary.csv"), index=False)

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = list(per_arm_patch)
    colors = plt.cm.tab10(np.arange(len(tags)))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for j, t in enumerate(tags):
        d = per_arm_patch[t]
        r = d.n_fake / d.n_real.clip(lower=1)
        bins = np.array([0, 15, 25, 33, 45, 56, 70, 90, 120, 250])
        idx = np.digitize(d.n_real, bins) - 1
        xs, ys = [], []
        for k in range(len(bins) - 1):
            mk = idx == k
            if mk.sum() >= 5:
                xs.append(d.n_real[mk].median()); ys.append(np.median(r[mk]))
        ax[0].plot(xs, ys, "-o", color=colors[j], label=t, ms=4)
    ax[0].axhline(1, color="k", lw=0.8, ls="--"); ax[0].set_xlabel("real count (StarDist)"); ax[0].set_ylabel("median fake/real")
    ax[0].set_title("fake/real ratio vs real count (val, mean of last checkpoints)"); ax[0].legend()
    w = 0.8 / len(tags)
    for j, t in enumerate(tags):
        d = per_arm_patch[t]; b = cl.density_bucket(d.n_real.values); r = d.n_fake / d.n_real.clip(lower=1)
        data = [np.clip(r[b == k], 0, 3) for k in cl.BUCKET_NAMES]
        vp = ax[1].violinplot(data, np.arange(3) + (j - (len(tags) - 1) / 2) * w, widths=w, showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(colors[j]); body.set_alpha(0.5)
    ax[1].set_xticks(range(3), cl.BUCKET_NAMES); ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_ylabel("fake/real (clipped at 3)"); ax[1].set_title("per-bucket distributions")
    for j, t in enumerate(tags):
        h = pd.read_csv(os.path.join(_script_dir, "outputs", "runs", t, "loss_history.csv"))
        ax[2].plot(h.epoch, h.loss_D, color=colors[j], label=f"{t} loss_D")
        ax[2].plot(h.epoch, h.loss_adv, color=colors[j], ls="--")
    ax[2].set_xlabel("epoch"); ax[2].set_title("loss_D (solid), loss_adv (dashed)"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "screening_summary.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 4.2), squeeze=False)
    for j, t in enumerate(tags):
        d = per_arm_patch[t]; b = cl.density_bucket(d.n_real.values)
        for k, c in zip(cl.BUCKET_NAMES, ("tab:green", "tab:blue", "tab:red")):
            ax[0, j].scatter(d.n_real[b == k], d.n_fake[b == k], s=5, alpha=0.5, c=c, label=k)
        mx = max(d.n_real.max(), d.n_fake.max()); ax[0, j].plot([0, mx], [0, mx], "k--", lw=0.8)
        ax[0, j].set_title(t); ax[0, j].set_xlabel("real"); ax[0, j].set_ylabel("fake (StarDist)")
    ax[0, 0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "screening_scatter.png"), dpi=100); plt.close(fig)
    print(f"\nwrote {OUT_DIR}")
