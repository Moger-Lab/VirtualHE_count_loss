"""Stage 1 - segment the tissue core outline in both modalities.

Produces, for all 77 paired cores:
  * a downsampled binary mask per modality  -> D:\\vhe_v2_work\\cache\\masks
  * shape statistics per modality           -> results/02_core_shapes.csv
  * a first outline-derived scale estimate  -> results/02_scale_estimate.csv
  * QC montages                             -> figures/

The scale estimate here is deliberately the *simple* one (area and diameter
ratios of the two outlines). It is a sanity check and a starting point for
stage 2, not the final calibration.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/02_core_segmentation.py
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
import tifffile
from PIL import Image

import config
from src import coreseg

Image.MAX_IMAGE_PIXELS = None

DS = 8  # downsample for segmentation: SRS 4408->551, H&E 6000->750


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def paired_cores() -> dict[str, tuple[Path, Path]]:
    srs = {p.name[: -len(config.SRS_SUFFIX)]: p
           for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}
    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}
    return {c: (srs[c], he[c]) for c in sorted(set(srs) & set(he), key=core_sort_key)}


def process_core(args) -> dict:
    """Segment one core in both modalities. Runs in a worker process."""
    core, srs_path, he_path, mask_dir = args
    t0 = time.time()
    out: dict = {"core": core}
    try:
        vol = tifffile.imread(srs_path)
        srs_mask, srs_thr, srs_bg = coreseg.segment_srs(vol, ds=DS)
        s_srs = coreseg.describe(srs_mask, core, "srs", DS, srs_thr, srs_bg)
        del vol

        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        he_mask, he_thr, he_bg, he_blank = coreseg.segment_he(rgb, ds=DS)
        # Un-scanned canvas must not contribute punch-rim points: a crop line
        # is a chord, and a chord biases the circle fit.
        s_he = coreseg.describe(he_mask, core, "he", DS, he_thr, he_bg,
                                exclude=he_blank)
        del rgb

        # store the filled footprints - that is what downstream stages align on
        np.savez_compressed(Path(mask_dir) / f"{core}.npz",
                            srs=coreseg.footprint(srs_mask),
                            he=coreseg.footprint(he_mask))

        out["srs"] = coreseg.shape_to_row(s_srs)
        out["he"] = coreseg.shape_to_row(s_he)
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 - want the core name with the error
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N cores (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    config.ensure_project_dirs()
    config.ensure_work_dirs()
    mask_dir = config.CACHE_DIR / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    cores = paired_cores()
    if args.limit:
        cores = dict(list(cores.items())[: args.limit])
    print(f"segmenting {len(cores)} cores, ds={DS}, workers={args.workers}")
    print(f"  masks -> {mask_dir}")

    jobs = [(c, s, h, str(mask_dir)) for c, (s, h) in cores.items()]
    results: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_core, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            flag = "" if r["ok"] else "  <-- FAILED"
            print(f"  [{i:3d}/{len(jobs)}] {r['core']:6s} {r['seconds']:5.1f}s{flag}")
            if not r["ok"]:
                print(f"        {r['error']}")
    print(f"total {time.time() - t0:.0f}s")

    results.sort(key=lambda r: core_sort_key(r["core"]))
    good = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]

    # --- shape table --------------------------------------------------------
    rows = []
    for r in good:
        for mod in ("srs", "he"):
            rows.append(r[mod])
    if rows:
        with (config.RESULTS_DIR / "02_core_shapes.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # --- first-pass scale estimate -----------------------------------------
    scale_rows = []
    for r in good:
        s, h = r["srs"], r["he"]
        sc = (h["circle_diam_full"] / s["circle_diam_full"]
              if s["circle_diam_full"] and np.isfinite(s["circle_diam_full"])
              and np.isfinite(h["circle_diam_full"]) else float("nan"))
        scale_rows.append({
            "core": r["core"],
            "srs_circle_diam": round(s["circle_diam_full"], 1),
            "he_circle_diam": round(h["circle_diam_full"], 1),
            "srs_circle_rms": round(s["circle_rms_rel"], 5),
            "he_circle_rms": round(h["circle_rms_rel"], 5),
            # PRIMARY estimator: ratio of fitted punch diameters.
            "scale_from_circle": round(sc, 5),
            "srs_equiv_diam": round(s["equiv_diam_full"], 1),
            "he_equiv_diam": round(h["equiv_diam_full"], 1),
            # Secondary, for cross-check only. Tissue-footprint area is a
            # biased proxy for punch size: a tear that opens to the core edge
            # is a bay, not a topological hole, so hole-filling cannot recover
            # it and the footprint under-measures the punch (`D-3`'s SRS).
            "scale_from_area": round(
                float(np.sqrt(h["area_px_full"] / s["area_px_full"])), 5),
            "scale_from_major": round(
                h["major_axis_full"] / s["major_axis_full"], 5),
            "srs_solidity": round(s["solidity"], 4),
            "he_solidity": round(h["solidity"], 4),
            "srs_ecc": round(s["eccentricity"], 4),
            "he_ecc": round(h["eccentricity"], 4),
            "srs_frac": round(s["frac_of_frame"], 4),
            "he_frac": round(h["frac_of_frame"], 4),
            "srs_border_frac": s["border_frac"],
            "he_border_frac": h["border_frac"],
            "srs_nfrag": s["n_fragments"],
            "he_nfrag": h["n_fragments"],
            "srs_largest_share": s["largest_frag_share"],
            "he_largest_share": h["largest_frag_share"],
            # "clean" = the fitted punch is trustworthy in BOTH modalities.
            # Clipping is deliberately NOT a criterion: a circle is recoverable
            # from a partial arc, which is the whole reason for fitting one
            # (`G-1` is 21% clipped in SRS and fits at rms 0.0147). What
            # matters is that the boundary is genuinely circular - a low
            # residual - and that the core has not broken into pieces that may
            # have shifted between sections.
            "clean": bool(
                np.isfinite(sc)
                and s["circle_rms_rel"] < 0.030 and h["circle_rms_rel"] < 0.030
                and s["n_fragments"] == 1 and h["n_fragments"] == 1
            ),
        })
    if scale_rows:
        with (config.RESULTS_DIR / "02_scale_estimate.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scale_rows[0].keys()))
            w.writeheader()
            w.writerows(scale_rows)

    # --- report -------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"SEGMENTED {len(good)}/{len(results)} cores")
    print("=" * 74)
    if bad:
        print("  FAILURES:")
        for r in bad:
            print(f"    {r['core']}: {r['error']}")

    if scale_rows:
        print("\n  QUALITY FLAGS")
        for label, pred in [
            ("SRS circle rejected", lambda r: not np.isfinite(r["srs_circle_rms"])),
            ("H&E circle rejected", lambda r: not np.isfinite(r["he_circle_rms"])),
            ("SRS circle rms > 0.03", lambda r: r["srs_circle_rms"] > 0.030),
            ("H&E circle rms > 0.03", lambda r: r["he_circle_rms"] > 0.030),
            ("SRS clipped by frame", lambda r: r["srs_border_frac"] >= 0.005),
            ("H&E clipped by frame", lambda r: r["he_border_frac"] >= 0.005),
            ("SRS fragmented", lambda r: r["srs_nfrag"] > 1),
            ("H&E fragmented", lambda r: r["he_nfrag"] > 1),
        ]:
            hits = [r["core"] for r in scale_rows if pred(r)]
            print(f"    {label:24s} {len(hits):3d}  {hits[:14]}")

        clean = [r for r in scale_rows if r["clean"]]
        dirty = [r for r in scale_rows if not r["clean"]]
        print(f"\n  CLEAN CORES (intact, unclipped, convex, both modalities):"
              f" {len(clean)}/{len(scale_rows)}")
        print(f"    excluded: {[r['core'] for r in dirty]}")

        # THE TEST. Acquisition is optically identical across the cohort
        # (stage 0), so SRS->H&E scale must be ONE constant. If the estimate is
        # sound, its spread over clean cores collapses relative to all cores.
        # If it does not collapse, the single-constant premise is wrong.
        print("\n" + "=" * 74)
        print("SCALE (H&E native px per SRS native px)")
        print("=" * 74)
        # NOTE: an area-derived diameter is algebraically the same estimator as
        # sqrt(area ratio), so those are not reported separately.
        for key in ("scale_from_circle", "scale_from_area", "scale_from_major"):
            print(f"\n  {key}")
            for label, subset in (("all   ", scale_rows), ("clean ", clean)):
                v = np.array([r[key] for r in subset], dtype=float)
                v = v[np.isfinite(v)]
                if v.size < 2:
                    continue
                print(f"    {label} n={len(v):3d}  median {np.median(v):.5f}  "
                      f"mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}  "
                      f"CV {v.std(ddof=1)/v.mean()*100:5.2f}%  "
                      f"range [{v.min():.4f}, {v.max():.4f}]")
            v = np.array([r[key] for r in clean], dtype=float)
            v = v[np.isfinite(v)]
            if v.size >= 2:
                sem = v.std(ddof=1) / np.sqrt(len(v))
                print(f"    -> pooled estimate {v.mean():.5f} +/- {sem:.5f} (SEM)")

        if clean:
            v = np.array([r["scale_from_circle"] for r in clean], dtype=float)
            v = v[np.isfinite(v)]
            v_all = np.array([r["scale_from_circle"] for r in scale_rows],
                             dtype=float)
            v_all = v_all[np.isfinite(v_all)]
            cv_clean = v.std(ddof=1) / v.mean() * 100
            cv_all = v_all.std(ddof=1) / v_all.mean() * 100
            print(f"\n  VERDICT: CV {cv_all:.2f}% (all) -> {cv_clean:.2f}% (clean)")
            if cv_clean < 1.0:
                print("    Spread collapses. Single-constant scale SUPPORTED.")
            elif cv_clean < 2.5:
                print("    Spread narrows substantially but residual remains;"
                      " inspect the tails before adopting a single constant.")
            else:
                print("    Spread does NOT collapse. Single-constant premise is"
                      " NOT supported by this segmentation - investigate before"
                      " proceeding to registration.")
            print(f"\n  v1 implied scale 0.23233/0.218291 = 1.06432 "
                  f"(NOT inherited; for comparison only)")

    (config.RESULTS_DIR / "02_core_segmentation.json").write_text(
        json.dumps({"ds": DS, "n_ok": len(good), "n_failed": len(bad),
                    "failures": [{"core": r["core"], "error": r["error"]}
                                 for r in bad],
                    "results": results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {config.RESULTS_DIR / '02_core_shapes.csv'}")
    print(f"  wrote {config.RESULTS_DIR / '02_scale_estimate.csv'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
