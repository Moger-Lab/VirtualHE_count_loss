"""
s02d -- counter acceptance tests (Step 2, CHECKPOINT A).

A. Validation (v2 fold 0, disjoint eval tiling, 496 patches, 11 cores):
     count Pearson r >= 0.95 versus StarDist
     per-bucket median count ratio within [0.95, 1.05]
B. Transfer to generator output: counter on Model 3's fake H (non-aug ep69,
   OLD registration val + test, cached by s01) versus StarDist on the
   corresponding fake RGB. Per bucket (real-count terciles), agreement
   within about 10%. Also counter on the real H of the same old patches
   versus StarDist real (registration-independent sanity check).
C. Synthetic: two touching nuclei count about 2, while the H-positive area
   fraction cannot distinguish them from one large nucleus of equal area.

Writes outputs/s02_counter/acceptance.json and figures.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import countlib as cl
import countnet as cn

OUT_DIR = os.path.join(_script_dir, "outputs", "s02_counter")
S01_CACHE = os.path.join(cl.WORK_DIR, "s01_model3_fakes")


def bucket_table(pred, ref, real_for_bucket):
    b = cl.density_bucket(real_for_bucket)
    rows = []
    for k in cl.BUCKET_NAMES + ("all",):
        m = np.ones(len(b), bool) if k == "all" else (b == k)
        rows.append(dict(bucket=k, n=int(m.sum()),
                         median_ratio=float(np.median(pred[m] / np.maximum(ref[m], 1))),
                         ratio_of_means=float(pred[m].mean() / max(ref[m].mean(), 1e-9)),
                         mean_pred=float(pred[m].mean()), mean_ref=float(ref[m].mean())))
    return pd.DataFrame(rows)


def run_counter(net, arrays, dev, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(arrays), bs):
            x = torch.from_numpy(np.stack(arrays[i:i + bs]).astype(np.float32))[:, None].to(dev)
            out.append(net(x).sum(dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


def part_a(net, dev, calib, res):
    from torch.utils.data import DataLoader
    sd = pd.read_csv(os.path.join(OUT_DIR, "stardist_per_patch.csv"))
    va = sd[(sd.split == "val") & (sd.role == "eval")].reset_index(drop=True)
    ck = torch.load(os.path.join(cl.WORK_DIR, "counter", "counter_best.pt"), map_location="cpu", weights_only=False)
    ds = cn.CounterDataset(va, calib, ck["sigma"], augment=False)
    dl = DataLoader(ds, batch_size=16, num_workers=6)
    pred, true = [], []
    with torch.no_grad():
        for b in dl:
            pred.append(net(b["h"].to(dev)).sum(dim=(1, 2, 3)).cpu().numpy()); true.append(b["count"].numpy())
    pred, true = np.concatenate(pred), np.concatenate(true)
    r = float(np.corrcoef(pred, true)[0, 1])
    t = bucket_table(pred, true, true)
    print("\n[A] validation, counter vs StarDist on real H&E (v2 fold 0):")
    print(f"    Pearson r = {r:.4f}  (criterion >= 0.95: {'PASS' if r >= 0.95 else 'FAIL'})")
    print(t.to_string(index=False))
    ok_b = bool(t[t.bucket != "all"].median_ratio.between(0.95, 1.05).all())
    print(f"    per-bucket median ratio within [0.95, 1.05]: {'PASS' if ok_b else 'FAIL'}")
    res["A"] = dict(pearson_r=r, pass_r=r >= 0.95, pass_buckets=ok_b, table=t.to_dict("records"))
    va = va.assign(counter=pred)
    va.to_csv(os.path.join(OUT_DIR, "acceptance_val_per_patch.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=6, alpha=0.5); mx = max(true.max(), pred.max())
    ax.plot([0, mx], [0, mx], "k--", lw=1); ax.set_xlabel("StarDist count (real)"); ax.set_ylabel("counter count")
    ax.set_title(f"val (fold 0), r = {r:.3f}"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acceptance_val_scatter.png"), dpi=110); plt.close(fig)


def part_b(net, dev, res):
    per = pd.read_csv(os.path.join(_script_dir, "outputs", "s01_diagnostic", "per_patch.csv"))
    fake_h, real_h, n_fake_sd, n_real_sd = [], [], [], []
    for pid in per.patch_id:
        z = np.load(os.path.join(S01_CACHE, f"{pid}.npz"))
        fake_h.append(z["fake_h"]); real_h.append(z["real_h"])
    per["counter_fake"] = run_counter(net, fake_h, dev)
    per["counter_real"] = run_counter(net, real_h, dev)
    tf_ = bucket_table(per.counter_fake.values, per.n_fake.values.astype(float), per.n_real.values)
    tr_ = bucket_table(per.counter_real.values, per.n_real.values.astype(float), per.n_real.values)
    r_f = float(np.corrcoef(per.counter_fake, per.n_fake)[0, 1])
    r_r = float(np.corrcoef(per.counter_real, per.n_real)[0, 1])
    print("\n[B] Model 3 fake H (non-aug ep69, old registration val+test): counter vs StarDist on fake RGB")
    print(f"    Pearson r = {r_f:.4f}")
    print(tf_.to_string(index=False))
    ok = bool(tf_[tf_.bucket != "all"].ratio_of_means.between(0.9, 1.1).all()
              and tf_[tf_.bucket != "all"].median_ratio.between(0.9, 1.1).all())
    print(f"    per-bucket agreement within 10% (median ratio and ratio of means): {'PASS' if ok else 'FAIL'}")
    print(f"\n    same patches, REAL H: counter vs StarDist real, r = {r_r:.4f}")
    print(tr_.to_string(index=False))
    res["B"] = dict(pearson_r_fake=r_f, pass_fake=ok, table_fake=tf_.to_dict("records"),
                    pearson_r_real_old=r_r, table_real_old=tr_.to_dict("records"))
    per.to_csv(os.path.join(OUT_DIR, "acceptance_model3_per_patch.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, x, y, t in ((ax[0], per.n_fake, per.counter_fake, f"Model 3 FAKE H, r={r_f:.3f}"),
                       (ax[1], per.n_real, per.counter_real, f"old-registration REAL H, r={r_r:.3f}")):
        a.scatter(x, y, s=4, alpha=0.4); mx = max(x.max(), y.max()); a.plot([0, mx], [0, mx], "k--", lw=1)
        a.set_xlabel("StarDist count on RGB"); a.set_ylabel("counter count on H"); a.set_title(t)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "acceptance_model3_scatter.png"), dpi=110); plt.close(fig)


def h_levels(calib, n=20):
    """Nucleus and background H levels in the calibrated [-1, 1] space,
    measured on real training patches around StarDist centroids."""
    sd = pd.read_csv(os.path.join(OUT_DIR, "stardist_per_patch.csv"))
    tr = sd[(sd.split == "train") & (sd.role == "train") & (sd.n_sd > 20)].sample(n, random_state=0)
    nuc, bg = [], []
    for r in tr.itertuples():
        _, he = cl.v2_patch_native(r.core, r.y0, r.x0, 441, channels=[2])
        h = cn.v1_common.rgb_to_he(cl.resize_rgb(he), calib)[0]
        cy, cx, _ = cl.stardist_cache()[r.patch_uid]
        yy, xx = np.mgrid[0:512, 0:512]
        near = np.zeros((512, 512), bool)
        for y, x in zip(cy, cx):
            near |= (yy - y) ** 2 + (xx - x) ** 2 < 25 ** 2
        nuc.append(h[np.clip(cy.astype(int), 0, 511), np.clip(cx.astype(int), 0, 511)])
        bg.append(h[~near])
    return float(np.median(np.concatenate(nuc))), float(np.median(np.concatenate(bg)))


def part_c(net, dev, calib, thr, res):
    ck = torch.load(os.path.join(cl.WORK_DIR, "counter", "counter_best.pt"), map_location="cpu", weights_only=False)
    r_eq = ck["sigma"] * 2.0
    h_nuc, h_bg = h_levels(calib)
    print(f"\n[C] synthetic: nucleus H {h_nuc:+.3f}, background H {h_bg:+.3f}, radius {r_eq:.1f} px")
    rng = np.random.default_rng(0)

    def canvas(blobs):
        yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
        m = np.zeros((512, 512), np.float32)
        for (cy, cx, rad) in blobs:
            m = np.maximum(m, (((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2).astype(np.float32))
        m = gaussian_filter(m, 1.5)
        h = h_bg + (h_nuc - h_bg) * m + rng.normal(0, 0.02, m.shape)
        return np.clip(h, -1, 1).astype(np.float32)

    r = r_eq
    configs = {
        "one": lambda c: [(c[0], c[1], r)],
        "two_touching": lambda c: [(c[0], c[1] - r, r), (c[0], c[1] + r, r)],
        "one_large_equal_area": lambda c: [(c[0], c[1], r * np.sqrt(2))],
        "two_overlapping_20pct": lambda c: [(c[0], c[1] - 0.8 * r, r), (c[0], c[1] + 0.8 * r, r)],
    }
    centres = [(y, x) for y in (96, 256, 416) for x in (96, 256, 416)]
    out = {}
    for name, f in configs.items():
        counts, fracs = [], []
        for c in centres:
            h = canvas(f(c))
            with torch.no_grad():
                counts.append(net(torch.from_numpy(h)[None, None].to(dev)).sum().item())
            fracs.append(float((h >= thr).mean()))
        out[name] = dict(count_mean=float(np.mean(counts)), count_sd=float(np.std(counts)),
                         hpos_area_fraction=float(np.mean(fracs)))
        print(f"    {name:24s} counter {np.mean(counts):.2f} +/- {np.std(counts):.2f}   H+ area fraction {np.mean(fracs):.5f}")
    two, big = out["two_touching"], out["one_large_equal_area"]
    area_same = abs(two["hpos_area_fraction"] - big["hpos_area_fraction"]) / two["hpos_area_fraction"] < 0.05
    ok = (1.6 <= two["count_mean"] <= 2.4) and (big["count_mean"] < two["count_mean"] - 0.5) and area_same
    print(f"    two touching ~2 ({two['count_mean']:.2f}), distinct from one equal-area blob ({big['count_mean']:.2f}); "
          f"area fraction differs by {100 * abs(two['hpos_area_fraction'] - big['hpos_area_fraction']) / two['hpos_area_fraction']:.1f}%: "
          f"{'PASS' if ok else 'FAIL'}")
    res["C"] = dict(configs=out, pass_=ok, h_nuc=h_nuc, h_bg=h_bg, radius=r_eq)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    for a, (name, f) in zip(ax, configs.items()):
        h = canvas(f((256, 256)))[176:336, 176:336]
        a.imshow(h, cmap="magma", vmin=-1, vmax=1); a.set_title(f"{name}\ncounter {out[name]['count_mean']:.2f}, "
                                                               f"H+ {out[name]['hpos_area_fraction'] * 512 * 512 / 9:.0f} px", fontsize=9)
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "acceptance_synthetic_blobs.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    dev = torch.device("cuda")
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    net = cn.load_counter(os.path.join(cl.WORK_DIR, "counter", "counter_best.pt"), dev)
    res = {}
    part_a(net, dev, calib, res)
    part_b(net, dev, res)
    part_c(net, dev, calib, float(calib.nuclear_h_threshold), res)
    with open(os.path.join(OUT_DIR, "acceptance.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    verdict = res["A"]["pass_r"] and res["A"]["pass_buckets"] and res["B"]["pass_fake"] and res["C"]["pass_"]
    print(f"\nCHECKPOINT A overall: {'ALL PASS' if verdict else 'NOT ALL PASS'}")
