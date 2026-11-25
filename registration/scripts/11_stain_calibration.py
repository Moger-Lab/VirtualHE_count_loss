"""Stage 4 - fit the H&E stain matrix to this cohort and set global ranges.

Produces:
  * one cohort stain matrix, fitted from pooled tissue pixels
  * per-core stain matrices, as a check on how homogeneous the cohort is
  * GLOBAL haematoxylin/eosin concentration ranges for normalisation
  * a stain-quality statistic per core

Normalisation ranges are global by construction. HANDOVER.md section 7 records
that per-patch percentile normalisation destroys absolute level - a dense
tumour patch and a sparse stromal patch each normalised to their own
percentiles come out equally stained, erasing the density signal the model must
learn to condition on.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/11_stain_calibration.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from skimage.transform import resize

import config
from src import coreseg, stain

Image.MAX_IMAGE_PIXELS = None

DS = 4
BETA = 0.15                 # OD floor separating stained tissue from glass
PIX_PER_CORE = 60000        # pooled sample size per core

# Robust percentile for the angular extremes of the stain cone.
#
# Chosen on deconvolution quality, not on resemblance to published vectors.
# Sweeping alpha over 4.2M pooled pixels (scratch analysis, 2026-09-22):
#
#   alpha    separation   condition   % clipped
#   0.01        38.5 deg       8.18       0.01
#   0.05        33.3 deg      11.15       0.05
#   1.00        24.3 deg      21.59       1.00
#
# Reconstruction residual is identical at every alpha - it is the component
# orthogonal to the SVD plane and so cannot depend on where the cone edges sit.
# That leaves conditioning and clipping, and both favour a small alpha: a
# narrow cone makes the unmixing ill-posed and clips pixels lying outside it to
# zero, and that clipping is not uniform - it removes the extremes, which are
# the densest nuclei, precisely the signal this project exists to preserve.
#
# alpha = 0.01 keeps each extreme supported by ~420 pixels, so the vectors are
# not defined by outliers, and recovers a 38.5 deg H-E separation against the
# 39.2 deg implied by the Ruifrok reference - independent corroboration that
# the geometry is physically sensible.
ALPHA = 0.01
N_BOOTSTRAP = 40


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def sample_core(args) -> dict:
    """Sample tissue OD pixels from one core, and fit its own stain matrix."""
    core, he_path, cy, cx, r, seed = args
    t0 = time.time()
    try:
        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        small = resize(rgb.astype(np.float32), (h // DS, w // DS, 3), order=1,
                       anti_aliasing=True, preserve_range=True)
        del rgb

        gy, gx = np.mgrid[0:small.shape[0], 0:small.shape[1]]
        disc = ((gy - cy / DS) ** 2 + (gx - cx / DS) ** 2) <= (0.95 * r / DS) ** 2
        disc &= ~coreseg.blank_canvas(small)      # never sample un-scanned canvas

        od = stain.tissue_od_pixels(small, beta=BETA, mask=disc)
        if od.shape[0] < 2000:
            raise ValueError(f"only {od.shape[0]} tissue pixels")

        rng = np.random.default_rng(seed)
        idx = rng.choice(od.shape[0], min(PIX_PER_CORE, od.shape[0]),
                         replace=False)
        sample = od[idx]

        own = stain.fit_macenko(sample, beta=BETA, alpha=ALPHA)
        return {"core": core, "ok": True, "seconds": round(time.time() - t0, 1),
                "sample": sample.astype(np.float32),
                "h": own.h.tolist(), "e": own.e.tolist(),
                "n_tissue_px": int(od.shape[0])}
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    config.ensure_project_dirs()
    config.ensure_work_dirs()

    tiers = json.loads(
        (config.RESULTS_DIR / "10_cohort_tiers.json").read_text(encoding="utf-8"))
    # Fit on the training pool only. The stain matrix is a parameter of the
    # pipeline, so fitting it on cores reserved for evaluation would leak.
    cores = sorted(tiers["train"], key=core_sort_key)

    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row
    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}

    jobs = []
    for i, c in enumerate(cores):
        h = shapes[c]["he"]
        jobs.append((c, he[c], float(h["circle_cy_full"]),
                     float(h["circle_cx_full"]),
                     float(h["circle_diam_full"]) / 2.0, 1000 + i))

    print(f"fitting stain matrix on {len(jobs)} training-pool cores "
          f"(eval cores held out)")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(sample_core, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"  {r['core']:6s} FAILED {r['error']}")
    print(f"total {time.time() - t0:.0f}s")

    good = [r for r in results if r["ok"]]
    good.sort(key=lambda r: core_sort_key(r["core"]))
    pooled = np.concatenate([r["sample"] for r in good], axis=0)
    cohort = stain.fit_macenko(pooled, beta=BETA, alpha=ALPHA)

    print("\n" + "=" * 74)
    print("COHORT STAIN MATRIX")
    print("=" * 74)
    print(f"  fitted from {cohort.n_pixels:,} tissue pixels over {len(good)} cores")
    print(f"    H = [{cohort.h[0]:.4f}, {cohort.h[1]:.4f}, {cohort.h[2]:.4f}]")
    print(f"    E = [{cohort.e[0]:.4f}, {cohort.e[1]:.4f}, {cohort.e[2]:.4f}]")
    print(f"  Ruifrok reference (NOT used, comparison only)")
    print(f"    H = [{stain.RUIFROK_H[0]:.4f}, {stain.RUIFROK_H[1]:.4f},"
          f" {stain.RUIFROK_H[2]:.4f}]")
    print(f"    E = [{stain.RUIFROK_E[0]:.4f}, {stain.RUIFROK_E[1]:.4f},"
          f" {stain.RUIFROK_E[2]:.4f}]")
    dh, de = cohort.angle_to(stain.RUIFROK_H, stain.RUIFROK_E)
    print(f"  angular deviation from Ruifrok: H {dh:.1f} deg, E {de:.1f} deg")
    sep = np.degrees(np.arccos(np.clip(cohort.h @ cohort.e, -1, 1)))
    ruif_sep = np.degrees(np.arccos(
        stain.RUIFROK_H @ stain.RUIFROK_E
        / (np.linalg.norm(stain.RUIFROK_H) * np.linalg.norm(stain.RUIFROK_E))))
    print(f"  separation between the two fitted stains: {sep:.1f} deg "
          f"(Ruifrok implies {ruif_sep:.1f} deg)")

    # --- bootstrap stability -----------------------------------------------
    # alpha = 0.01 leans on the tails, so confirm the vectors are a property of
    # the cohort rather than of the particular pixels sampled.
    rng = np.random.default_rng(12345)
    bh, be = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, pooled.shape[0], pooled.shape[0] // 4)
        try:
            s = stain.fit_macenko(pooled[idx], beta=BETA, alpha=ALPHA)
        except ValueError:
            continue
        bh.append(np.degrees(np.arccos(np.clip(s.h @ cohort.h, -1, 1))))
        be.append(np.degrees(np.arccos(np.clip(s.e @ cohort.e, -1, 1))))
    bh, be = np.array(bh), np.array(be)
    print(f"\n  bootstrap ({len(bh)} resamples at 25% of pixels):")
    print(f"    H vector wobble: median {np.median(bh):.2f} deg, "
          f"p95 {np.percentile(bh, 95):.2f} deg")
    print(f"    E vector wobble: median {np.median(be):.2f} deg, "
          f"p95 {np.percentile(be, 95):.2f} deg")

    # --- per-core spread ----------------------------------------------------
    hs = np.array([r["h"] for r in good])
    es = np.array([r["e"] for r in good])
    ah = np.array([np.degrees(np.arccos(np.clip(v @ cohort.h, -1, 1)))
                   for v in hs])
    ae = np.array([np.degrees(np.arccos(np.clip(v @ cohort.e, -1, 1)))
                   for v in es])
    print("\n" + "=" * 74)
    print("PER-CORE STAIN VECTORS vs THE COHORT FIT")
    print("=" * 74)
    print(f"  H deviation: median {np.median(ah):5.2f} deg  p90 {np.percentile(ah,90):5.2f}"
          f"  max {ah.max():5.2f} ({good[int(ah.argmax())]['core']})")
    print(f"  E deviation: median {np.median(ae):5.2f} deg  p90 {np.percentile(ae,90):5.2f}"
          f"  max {ae.max():5.2f} ({good[int(ae.argmax())]['core']})")
    outliers = sorted({good[i]["core"] for i in np.argsort(-(ah + ae))[:6]},
                      key=core_sort_key)
    print(f"  largest combined deviation: {outliers}")

    # --- GLOBAL concentration ranges ---------------------------------------
    conc = pooled @ np.linalg.pinv(cohort.matrix)
    conc = np.clip(conc, 0, None)
    print("\n" + "=" * 74)
    print("GLOBAL CONCENTRATION RANGES (applied unchanged to every patch)")
    print("=" * 74)
    ranges = {}
    for i, name in enumerate(("haematoxylin", "eosin")):
        c = conc[:, i]
        qs = {f"p{p}": float(np.percentile(c, p))
              for p in (0.1, 1, 5, 50, 95, 99, 99.5, 99.9)}
        ranges[name] = qs
        print(f"  {name:13s} median {qs['p50']:.4f}   p99 {qs['p99']:.4f}   "
              f"p99.9 {qs['p99.9']:.4f}   max {c.max():.4f}")
    # p99.9 rather than the max: robust to saturated pixels and dust, while
    # still spanning genuinely dense nuclei.
    norm = {k: v["p99.9"] for k, v in ranges.items()}
    print(f"\n  normalisation scale (p99.9): H {norm['haematoxylin']:.4f}, "
          f"E {norm['eosin']:.4f}")

    out = {
        "fitted_on": {"n_cores": len(good), "cores": [r["core"] for r in good],
                      "n_pixels": int(cohort.n_pixels),
                      "note": "training pool only; eval cores held out"},
        "beta": BETA, "alpha": ALPHA, "ds": DS,
        "H": cohort.h.tolist(), "E": cohort.e.tolist(),
        "deviation_from_ruifrok_deg": {"H": dh, "E": de},
        "stain_separation_deg": float(sep),
        "ruifrok_separation_deg": float(ruif_sep),
        "bootstrap_wobble_deg": {
            "n": int(len(bh)),
            "H_median": float(np.median(bh)), "H_p95": float(np.percentile(bh, 95)),
            "E_median": float(np.median(be)), "E_p95": float(np.percentile(be, 95))},
        "global_ranges": ranges,
        "normalisation_p999": norm,
        "per_core": [{"core": r["core"], "H": r["h"], "E": r["e"],
                      "n_tissue_px": r["n_tissue_px"],
                      "dev_H_deg": float(ah[i]), "dev_E_deg": float(ae[i])}
                     for i, r in enumerate(good)],
    }
    (config.RESULTS_DIR / "11_stain_calibration.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {config.RESULTS_DIR / '11_stain_calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
