"""Stage 4b - detect cores with genuinely defective H&E staining.

HANDOVER.md section 7 names three: `C-7`, `H-1`, `B-7`, at H-area per detected
nucleus of 0.27 / 0.42 / 0.44 of the cohort median, and specifies the metric:

  **H-area fraction per StarDist-detected nucleus.**

StarDist reads the RGB and is robust to stain intensity; the deconvolved
haematoxylin channel is not. Dividing one by the other therefore isolates
*stain strength* from *tissue composition*. The brief records that two simpler
statistics - mean per-patch H p99, and its q90 - both mis-identified the
affected set, because each conflates stain quality with how much nuclear
material is present.

This is an independent re-derivation, not an inheritance: the three cores are
named in the brief but the detection is run from scratch on this cohort's own
stain matrix, and the result is compared against them afterwards.

Also flagged here, separately: `G-1` and `G-9` are stroma-rich with genuinely
low nuclear density and normal staining, and must NOT be gated out
(HANDOVER.md section 7). A correct metric distinguishes them from the
defective set; that distinction is the real test of it.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/12_stain_quality.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config

PATCH_UM = 100.0
RENDER_PX = 512
SRS_UM_PER_PX = config.SRS_UM_PER_PX
COHORT_SCALE = config.COHORT_SCALE
HE_UM_PER_PX = config.HE_UM_PER_PX

MIN_TISSUE_FRAC = 0.50
MAX_BLANK_FRAC = 0.02
DISC_FRAC = 0.90
# A pixel counts toward "H area" if its haematoxylin concentration exceeds this
# fraction of the global p99.9 scale. Referenced to the GLOBAL range, never to
# the patch - a per-patch reference would normalise away the very differences
# being measured.
H_AREA_THRESH_FRAC = 0.35


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def _init_worker():
    n = str(config.STARDIST_THREADS_PER_PROC)
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["TF_NUM_INTRAOP_THREADS"] = n
    os.environ["TF_NUM_INTEROP_THREADS"] = n
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from stardist.models import StarDist2D
        _MODEL = StarDist2D.from_pretrained(config.STARDIST_MODEL)
    return _MODEL


def process_core(args) -> dict:
    core, he_path, cy, cx, r, sm_h, sm_e, h_scale = args
    t0 = time.time()
    try:
        from csbdeep.utils import normalize
        from PIL import Image
        from skimage.transform import resize
        from src import stain

        Image.MAX_IMAGE_PIXELS = None
        model = get_model()
        sm = stain.StainMatrix(h=np.array(sm_h), e=np.array(sm_e),
                               n_pixels=0, beta=0.15, alpha=0.01)

        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))

        patch_px = int(round(PATCH_UM / HE_UM_PER_PX))
        per_patch = []

        for y0 in np.arange(int(cy - r), int(cy + r) - patch_px + 1, patch_px):
            for x0 in np.arange(int(cx - r), int(cx + r) - patch_px + 1, patch_px):
                pcy, pcx = y0 + patch_px / 2, x0 + patch_px / 2
                if (pcy - cy) ** 2 + (pcx - cx) ** 2 > (DISC_FRAC * r) ** 2:
                    continue
                if y0 < 0 or x0 < 0 or y0 + patch_px > rgb.shape[0] \
                        or x0 + patch_px > rgb.shape[1]:
                    continue
                p = rgb[y0:y0 + patch_px, x0:x0 + patch_px]
                if (p.min(axis=-1) >= 250).mean() > MAX_BLANK_FRAC:
                    continue
                od = -np.log10(np.clip(p.astype(np.float32), 1, 255) / 255.0)
                if float((od.sum(axis=-1) > 0.15).mean()) < MIN_TISSUE_FRAC:
                    continue

                q = resize(p.astype(np.float32), (RENDER_PX, RENDER_PX, 3),
                           order=1, anti_aliasing=True, preserve_range=True)
                lab, _ = model.predict_instances(
                    normalize(q, *config.STARDIST_NORM_PERCENTILES, axis=(0, 1)),
                    verbose=False)
                ids, areas = np.unique(lab, return_counts=True)
                keep = (ids > 0) & (areas >= config.STARDIST_MIN_NUCLEUS_AREA_PX)
                n_nuc = int(keep.sum())
                if n_nuc < 3:
                    continue

                conc = stain.deconvolve(q, sm)
                h_area = float((conc[..., 0] > H_AREA_THRESH_FRAC * h_scale).mean())
                per_patch.append((h_area, n_nuc,
                                  float(np.percentile(conc[..., 0], 99))))

        if len(per_patch) < 5:
            raise ValueError(f"only {len(per_patch)} usable patches")
        a = np.array(per_patch)
        # nuclei per 512x512 render -> density, so the ratio is per-nucleus
        # H area regardless of how many nuclei a patch happens to contain
        h_per_nuc = a[:, 0] / (a[:, 1] / (RENDER_PX * RENDER_PX) * 1e4)
        return {"core": core, "ok": True, "seconds": round(time.time() - t0, 1),
                "n_patches": len(a),
                "h_area_frac": float(np.median(a[:, 0])),
                "nuclei_per_patch": float(np.median(a[:, 1])),
                "h_area_per_nucleus": float(np.median(h_per_nuc)),
                "h_p99": float(np.median(a[:, 2]))}
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    config.ensure_project_dirs()
    cal = json.loads(
        (config.RESULTS_DIR / "11_stain_calibration.json").read_text(encoding="utf-8"))
    h_scale = cal["normalisation_p999"]["haematoxylin"]

    tiers = json.loads(
        (config.RESULTS_DIR / "10_cohort_tiers.json").read_text(encoding="utf-8"))
    cores = sorted(set(tiers["train"]) | set(tiers["unpaired_only"]),
                   key=core_sort_key)

    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row
    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}

    jobs = [(c, he[c], float(shapes[c]["he"]["circle_cy_full"]),
             float(shapes[c]["he"]["circle_cx_full"]),
             float(shapes[c]["he"]["circle_diam_full"]) / 2.0,
             cal["H"], cal["E"], h_scale) for c in cores]

    print(f"stain quality on {len(jobs)} cores "
          f"(H-area threshold {H_AREA_THRESH_FRAC:.2f} x global p99.9 "
          f"= {H_AREA_THRESH_FRAC * h_scale:.4f})")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(process_core, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"  {r['core']:6s} FAILED {r['error']}")
            elif i % 20 == 0:
                print(f"  [{i:3d}/{len(jobs)}] ...")
    print(f"total {time.time() - t0:.0f}s")

    good = [r for r in results if r["ok"]]
    good.sort(key=lambda r: core_sort_key(r["core"]))
    vals = np.array([r["h_area_per_nucleus"] for r in good])
    med = float(np.median(vals))
    for r in good:
        r["ratio_to_median"] = round(r["h_area_per_nucleus"] / med, 4)

    print("\n" + "=" * 74)
    print("H-AREA PER StarDist NUCLEUS, relative to the cohort median")
    print("=" * 74)
    ranked = sorted(good, key=lambda r: r["ratio_to_median"])
    print(f"{'core':7s}{'ratio':>8s}{'h_area':>9s}{'nuc/patch':>11s}"
          f"{'h_p99':>8s}   note")
    for r in ranked[:12]:
        note = ""
        if r["core"] in config.V1_DEFECTIVE_HE_CORES:
            note = "<- v1 flagged DEFECTIVE"
        elif r["core"] in config.V1_STROMA_RICH_CORES:
            note = "<- v1: stroma-rich, NOT a defect"
        print(f"{r['core']:7s}{r['ratio_to_median']:8.3f}"
              f"{r['h_area_frac']:9.3f}{r['nuclei_per_patch']:11.1f}"
              f"{r['h_p99']:8.3f}   {note}")

    print("\n  ... highest ...")
    for r in ranked[-3:]:
        print(f"{r['core']:7s}{r['ratio_to_median']:8.3f}"
              f"{r['h_area_frac']:9.3f}{r['nuclei_per_patch']:11.1f}"
              f"{r['h_p99']:8.3f}")

    # --- comparison against v1 ---------------------------------------------
    print("\n" + "=" * 74)
    print("COMPARISON WITH v1")
    print("=" * 74)
    v1_ratios = {"C-7": 0.27, "H-1": 0.42, "B-7": 0.44}
    by = {r["core"]: r for r in good}
    for c, v1r in v1_ratios.items():
        here = by.get(c, {}).get("ratio_to_median")
        rank = ([r["core"] for r in ranked].index(c) + 1) if c in by else None
        print(f"  {c:5s} v1 {v1r:.2f}   this work "
              f"{here if here is None else f'{here:.3f}'}   "
              f"rank {rank}/{len(ranked)} (1 = worst)")
    for c in config.V1_STROMA_RICH_CORES:
        if c in by:
            rank = [r["core"] for r in ranked].index(c) + 1
            print(f"  {c:5s} stroma-rich control: ratio "
                  f"{by[c]['ratio_to_median']:.3f}, rank {rank}/{len(ranked)} "
                  f"- should NOT be near the bottom")

    with (config.RESULTS_DIR / "12_stain_quality.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(good[0].keys()))
        w.writeheader()
        w.writerows(good)
    print(f"\n  wrote {config.RESULTS_DIR / '12_stain_quality.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

