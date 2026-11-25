"""
s18d -- smoke-test report for the detector screen (any env with numpy,
pandas, scikit-image, matplotlib; run in vhe-detectors).

Reads s18a inputs (StarDist labels inside) and s18b/s18c label files.
Counts use the v1 rule (instances >= 10 px on the 512 grid) for every
detector. Writes to outputs\\s18_metric_screen\\smoke\\ (never overwrites):
  smoke_counts.csv       one row per patch x image x detector
  smoke_detector_summary.csv
  smoke_montage.png      contours on real / fake patches, one row per detector
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
IN_DIR = __import__("vhe_paths").work("s18_metric_screen", "smoke")
OUT = os.path.join(_script_dir, "outputs", "s18_metric_screen", "smoke")
MIN_AREA = 10
MODELS = ("CG", "CGM", "CGMA", "CGMC")


def n_inst(lab):
    if lab.max() == 0:
        return 0, np.array([])
    a = np.bincount(lab.ravel())[1:]
    a = a[a > 0]
    return int((a >= MIN_AREA).sum()), a[a >= MIN_AREA]


def slope(x, y):
    m = (x > 0) & (y > 0)
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]) if m.sum() > 2 else np.nan


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "smoke_counts.csv")):
        sys.exit(f"{OUT} already has results; refusing to overwrite")
    pt = pd.read_csv(os.path.join(IN_DIR, "patches.csv"))
    dets = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(IN_DIR, "det", "*__*.npz"))})
    rows = []
    for r in pt.itertuples():
        z = np.load(os.path.join(IN_DIR, f"{r.patch_uid}.npz"))
        imgs = ["real_rgb", "rt_rgb"] + [f"fake_rgb_{m}" for m in MODELS]
        for im in imgs:
            n, a = n_inst(z["sd_lab_" + im])
            rows.append(dict(patch_uid=r.patch_uid, core=r.core, bucket=r.bucket, image=im, detector="stardist",
                             n=n, med_area=float(np.median(a)) if n else np.nan))
        for d in dets:
            zd = np.load(os.path.join(IN_DIR, "det", f"{d}__{r.patch_uid}.npz"))
            for im in imgs:
                n, a = n_inst(zd[im])
                rows.append(dict(patch_uid=r.patch_uid, core=r.core, bucket=r.bucket, image=im, detector=d,
                                 n=n, med_area=float(np.median(a)) if n else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "smoke_counts.csv"), index=False)

    wide = df.pivot_table(index=["patch_uid", "image"], columns="detector", values="n").reset_index()
    timing = pd.concat([pd.read_csv(os.path.join(IN_DIR, "det", "timing.csv")),
                        pd.read_csv(os.path.join(IN_DIR, "timing_stardist.csv")).assign(detector="stardist")])
    summ = []
    for d in ["stardist"] + dets:
        real = wide[wide.image == "real_rgb"]; rt = wide[wide.image == "rt_rgb"]
        fake = wide[wide.image.str.startswith("fake")]
        rec = dict(detector=d)
        rec["real_r_vs_sd"] = float(np.corrcoef(real.stardist, real[d])[0, 1])
        rec["real_slope_vs_sd"] = slope(real.stardist.values, real[d].values)
        rec["real_median_ratio_vs_sd"] = float(np.median(real[d] / real.stardist.clip(lower=1)))
        rec["R1_roundtrip_ratio_median"] = float(np.median(rt[d].values / real[d].clip(lower=1).values))
        rec["fake_r_vs_sd"] = float(np.corrcoef(fake.stardist, fake[d])[0, 1])
        rec["fake_median_ratio_vs_sd"] = float(np.median(fake[d] / fake.stardist.clip(lower=1)))
        rec["fake_min_n"] = int(fake[d].min())
        rec["fake_frac_empty"] = float((fake[d] == 0).mean())
        rec["fake_frac_lt5"] = float((fake[d] < 5).mean())
        for m in MODELS:
            fm = wide[wide.image == f"fake_rgb_{m}"]
            rec[f"slope_fake_on_real_{m}"] = slope(real[d].values, fm[d].values)
        rec["sec_per_image_median"] = float(timing[timing.detector == d].sec.median())
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "smoke_detector_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(summ.T.to_string())

    # montage: 3 patches (sparse / mid / dense) x [real, fake CG, fake CGMC], one row block per detector
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    picks = [pt[pt.bucket == b].iloc[len(pt[pt.bucket == b]) // 2].patch_uid for b in ("low", "med", "high")]
    cols = [(u, im) for u in picks for im in ("real_rgb", "fake_rgb_CG", "fake_rgb_CGMC")]
    allds = ["stardist"] + dets
    fig, ax = plt.subplots(len(allds), len(cols), figsize=(2.2 * len(cols), 2.3 * len(allds)))
    for i, d in enumerate(allds):
        for j, (u, im) in enumerate(cols):
            z = np.load(os.path.join(IN_DIR, f"{u}.npz"))
            lab = z["sd_lab_" + im] if d == "stardist" else np.load(os.path.join(IN_DIR, "det", f"{d}__{u}.npz"))[im]
            rgb = z[im].copy(); b = find_boundaries(lab, mode="outer"); rgb[b] = (0, 255, 0)
            ax[i, j].imshow(rgb); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(f"{u.split('_eval')[0]} {im.replace('_rgb', '').replace('fake_', '')}", fontsize=8)
            if j == 0:
                ax[i, j].set_ylabel(d, fontsize=9)
            ax[i, j].text(5, 25, str(n_inst(lab)[0]), color="yellow", fontsize=9, weight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "smoke_montage.png"), dpi=110); plt.close(fig)
    print(f"wrote {OUT}")
