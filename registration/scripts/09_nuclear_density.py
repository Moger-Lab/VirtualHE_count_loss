"""Stage 3c - nuclear density of the REAL H&E, per core, by registration tier.

Purpose
-------
Decide empirically whether tier-B cores (correctly registered, low structural
correlation) cover nuclear density range that tier-A cores do not. The project's
target is recovering density *dynamic range*, so if tier B occupies range tier A
lacks, excluding it would truncate the training distribution at exactly the end
the model has to learn to discriminate. If tier B is redundant in density terms,
the case for including it is much weaker.

This is a "scan before you commit" check (HANDOVER.md section 8): cheap, run
before any training decision, not after.

Protocol
--------
Exactly v1's, so the numbers stay comparable (HANDOVER.md section 6):
StarDist `2D_versatile_he`, normalisation percentiles 1.0/99.8, minimum nucleus
area 10 px, 100 um patches rendered at 512 px.

Note this measures the REAL H&E only. No model is involved; nothing here
depends on registration being correct, which is what makes it a fair way to
compare tiers defined *by* registration quality.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/09_nuclear_density.py
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

MIN_TISSUE_FRAC = 0.50      # skip patches that are mostly empty slide
MAX_BLANK_FRAC = 0.02       # skip patches touching un-scanned canvas
DISC_FRAC = 0.90            # keep patch centres inside this fraction of radius


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def _init_worker():
    """Cap TF threading per process - StarDist is CPU-only on Windows."""
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
    core, he_path, cy, cx, r = args
    t0 = time.time()
    try:
        from csbdeep.utils import normalize
        from PIL import Image
        from skimage.transform import resize

        Image.MAX_IMAGE_PIXELS = None
        model = get_model()

        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))

        patch_px = int(round(PATCH_UM / HE_UM_PER_PX))
        counts, tissue_fracs, coords = [], [], []

        y0s = np.arange(int(cy - r), int(cy + r) - patch_px + 1, patch_px)
        x0s = np.arange(int(cx - r), int(cx + r) - patch_px + 1, patch_px)
        for y0 in y0s:
            for x0 in x0s:
                pcy, pcx = y0 + patch_px / 2, x0 + patch_px / 2
                if (pcy - cy) ** 2 + (pcx - cx) ** 2 > (DISC_FRAC * r) ** 2:
                    continue
                if y0 < 0 or x0 < 0 or y0 + patch_px > rgb.shape[0] \
                        or x0 + patch_px > rgb.shape[1]:
                    continue
                p = rgb[y0:y0 + patch_px, x0:x0 + patch_px]

                # reject un-scanned canvas and mostly-empty slide
                blank = (p.min(axis=-1) >= 250).mean()
                if blank > MAX_BLANK_FRAC:
                    continue
                od = -np.log10(np.clip(p.astype(np.float32), 1, 255) / 255.0)
                tissue = float((od.sum(axis=-1) > 0.15).mean())
                if tissue < MIN_TISSUE_FRAC:
                    continue

                q = resize(p.astype(np.float32), (RENDER_PX, RENDER_PX, 3),
                           order=1, anti_aliasing=True, preserve_range=True)
                lab, _ = model.predict_instances(
                    normalize(q, *config.STARDIST_NORM_PERCENTILES, axis=(0, 1)),
                    verbose=False)
                ids, areas = np.unique(lab, return_counts=True)
                keep = (ids > 0) & (areas >= config.STARDIST_MIN_NUCLEUS_AREA_PX)
                counts.append(int(keep.sum()))
                tissue_fracs.append(tissue)
                coords.append((int(y0), int(x0)))

        if not counts:
            raise ValueError("no admissible patches")
        c = np.array(counts, dtype=float)
        return {"core": core, "ok": True, "seconds": round(time.time() - t0, 1),
                "n_patches": len(c),
                "mean": float(c.mean()), "median": float(np.median(c)),
                "p10": float(np.percentile(c, 10)),
                "p90": float(np.percentile(c, 90)),
                "max": float(c.max()), "min": float(c.min()),
                "mean_tissue_frac": float(np.mean(tissue_fracs)),
                "counts": counts}
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 1)}


def load_tiers() -> dict[str, str]:
    """Canonical tiers from stage 10 - never re-derived here.

    This previously recomputed the tiering with its own copy of the rules,
    which then drifted when stage 10's criterion was corrected. One definition,
    one file.
    """
    rows = list(csv.DictReader(
        (config.RESULTS_DIR / "10_cohort_tiers.csv").open(encoding="utf-8")))
    return {r["core"]: r["tier"] for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tiers", nargs="+",
                    default=["A_good", "B_weak_signal"])
    args = ap.parse_args()

    config.ensure_project_dirs()
    tiers = load_tiers()
    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row
    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}

    cores = [c for c in sorted(tiers, key=core_sort_key)
             if tiers[c] in args.tiers and c in he and c in shapes]
    print(f"StarDist density on {len(cores)} cores, tiers {args.tiers}")
    print(f"  protocol: {config.STARDIST_MODEL}, norm "
          f"{config.STARDIST_NORM_PERCENTILES}, min area "
          f"{config.STARDIST_MIN_NUCLEUS_AREA_PX} px, "
          f"{PATCH_UM:.0f} um patches at {RENDER_PX} px "
          f"({int(round(PATCH_UM / HE_UM_PER_PX))} native px)")

    jobs = []
    for c in cores:
        h = shapes[c]["he"]
        jobs.append((c, he[c], float(h["circle_cy_full"]),
                     float(h["circle_cx_full"]),
                     float(h["circle_diam_full"]) / 2.0))

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(process_core, j): j[0] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if not r["ok"]:
                print(f"  [{done:3d}/{len(jobs)}] {r['core']:6s} FAILED "
                      f"{r['error']}")
            elif done % 10 == 0:
                print(f"  [{done:3d}/{len(jobs)}] ...")
    print(f"total {time.time() - t0:.0f}s")

    good = [r for r in results if r["ok"]]
    good.sort(key=lambda r: core_sort_key(r["core"]))
    for r in good:
        r["tier"] = tiers[r["core"]]

    rows = [{k: v for k, v in r.items() if k != "counts"} for r in good]
    with (config.RESULTS_DIR / "09_nuclear_density.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 74)
    print("NUCLEAR DENSITY BY TIER (real H&E, nuclei per 100 um patch)")
    print("=" * 74)
    by_tier = {}
    for r in good:
        by_tier.setdefault(r["tier"], []).extend(r["counts"])
    for tier in sorted(by_tier):
        c = np.array(by_tier[tier], dtype=float)
        ncore = sum(1 for r in good if r["tier"] == tier)
        print(f"\n  {tier}  ({ncore} cores, {len(c)} patches)")
        print(f"    mean {c.mean():6.1f}   median {np.median(c):6.1f}   "
              f"sd {c.std(ddof=1):6.1f}")
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f"      p{p:<3d} {np.percentile(c, p):7.1f}", end="")
            if p in (10, 50, 99):
                print()
        print(f"    min {c.min():.0f}   max {c.max():.0f}")

    if "A_good" in by_tier and "B_weak_signal" in by_tier:
        a = np.array(by_tier["A_good"], dtype=float)
        b = np.array(by_tier["B_weak_signal"], dtype=float)
        print("\n" + "=" * 74)
        print("DOES TIER B COVER RANGE TIER A DOES NOT?")
        print("=" * 74)
        print(f"  tier A range [{a.min():.0f}, {a.max():.0f}], "
              f"IQR [{np.percentile(a,25):.0f}, {np.percentile(a,75):.0f}]")
        print(f"  tier B range [{b.min():.0f}, {b.max():.0f}], "
              f"IQR [{np.percentile(b,25):.0f}, {np.percentile(b,75):.0f}]")
        lo_a = np.percentile(a, 5)
        frac_b_below = float((b < lo_a).mean())
        print(f"\n  tier A 5th percentile: {lo_a:.1f} nuclei/patch")
        print(f"  share of tier B patches below it: {frac_b_below*100:.1f}%")
        print(f"  share of tier A patches below it: 5.0% (by definition)")
        enrich = frac_b_below / 0.05 if frac_b_below else 0.0
        print(f"  -> tier B is {enrich:.2f}x enriched in the sparse tail")
        print(f"\n  per-core medians:")
        am = [r["median"] for r in good if r["tier"] == "A_good"]
        bm = [r["median"] for r in good if r["tier"] == "B_weak_signal"]
        print(f"    tier A: median-of-medians {np.median(am):.1f}, "
              f"range [{min(am):.0f}, {max(am):.0f}]")
        print(f"    tier B: median-of-medians {np.median(bm):.1f}, "
              f"range [{min(bm):.0f}, {max(bm):.0f}]")

    (config.RESULTS_DIR / "09_nuclear_density.json").write_text(
        json.dumps({"protocol": {"model": config.STARDIST_MODEL,
                                 "norm": config.STARDIST_NORM_PERCENTILES,
                                 "min_area": config.STARDIST_MIN_NUCLEUS_AREA_PX,
                                 "patch_um": PATCH_UM, "render_px": RENDER_PX},
                    "cores": good}, indent=2), encoding="utf-8")
    print(f"\n  wrote {config.RESULTS_DIR / '09_nuclear_density.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

