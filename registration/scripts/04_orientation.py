"""Stage 2 - determine the SRS/H&E mirror parity empirically, per core.

HANDOVER.md section 2: "Orientation is also unknown. v1 applied a horizontal
flip to the SRS. Determine flips empirically for both axes and both modalities."

The test
--------
For every core independently, find the best rotation aligning the H&E to the
SRS under each mirror parity, and record which parity wins and by how much.

A mirror is a *global* property of the imaging setup - one microscope, one
scanner, one slide-handling convention - so if a mirror really is present, the
same parity must win across essentially every core.

CORRECTED PREMISE: an earlier version of this script warned if the recovered
rotations were *concentrated*, on the reasoning that independent cores should
sit at arbitrary angles. That reasoning was wrong. TMA cores are punched into a
fixed grid on one slide, and that slide is loaded on each instrument in a
consistent orientation, so a shared rotation is exactly what a correct result
looks like. Concentration is corroboration, not a warning sign. What would
genuinely indicate a problem is a *uniform* rotation distribution, which would
mean the correlation is finding nothing.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/04_orientation.py
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
from scipy import ndimage as ndi
from skimage import filters
from skimage.transform import resize

import config
from src import coreseg, orient

Image.MAX_IMAGE_PIXELS = None

DS = 8
SMOOTH_SIGMA = 3.0   # at ds=8 this is ~24 native px ~ 5.5 um; further smoothing
                     # happens implicitly through the coarse polar grid


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def load_circles() -> dict[str, dict]:
    """Fitted punch circles from stage 1, keyed by core then modality."""
    path = config.RESULTS_DIR / "02_core_shapes.csv"
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["core"], {})[row["modality"]] = row
    return out


def srs_density(vol: np.ndarray, ds: int) -> np.ndarray:
    """Tissue-density map from CARS - tissue presence, not tissue type."""
    if vol.shape[0] != 4 and vol.shape[-1] == 4:
        vol = np.moveaxis(vol, -1, 0)
    ch = vol[coreseg.TISSUE_CHANNELS[0]]
    y, x = ch.shape
    s = resize(ch.astype(np.float32), (y // ds, x // ds),
               order=1, anti_aliasing=True, preserve_range=True)
    lo, hi = np.percentile(s, (1.0, 99.5))
    s = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)
    return filters.gaussian(s, sigma=SMOOTH_SIGMA)


def he_density(rgb: np.ndarray, ds: int) -> np.ndarray:
    """Tissue-density map from H&E optical density."""
    y, x = rgb.shape[:2]
    s = resize(rgb[..., :3].astype(np.float32), (y // ds, x // ds, 3),
               order=1, anti_aliasing=True, preserve_range=True)
    od = -np.log10(np.clip(s, 1.0, 255.0) / 255.0).sum(axis=-1)
    return filters.gaussian(od, sigma=SMOOTH_SIGMA)


def process_core(args) -> dict:
    core, srs_path, he_path, circ = args
    t0 = time.time()
    try:
        # circle params are stored in NATIVE pixels; polar sampling happens on
        # the ds-downsampled maps, so convert.
        s_cy = float(circ["srs"]["circle_cy_full"]) / DS
        s_cx = float(circ["srs"]["circle_cx_full"]) / DS
        s_r = float(circ["srs"]["circle_diam_full"]) / 2.0 / DS
        h_cy = float(circ["he"]["circle_cy_full"]) / DS
        h_cx = float(circ["he"]["circle_cx_full"]) / DS
        h_r = float(circ["he"]["circle_diam_full"]) / 2.0 / DS
        if not all(np.isfinite([s_cy, s_cx, s_r, h_cy, h_cx, h_r])):
            raise ValueError("non-finite circle parameters from stage 1")

        vol = tifffile.imread(srs_path)
        srs_d = srs_density(vol, DS)
        del vol
        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        he_d = he_density(rgb, DS)
        del rgb

        srs_polar = orient.to_polar(srs_d, s_cy, s_cx, s_r)
        he_polar = orient.to_polar(he_d, h_cy, h_cx, h_r)
        res = orient.determine(srs_polar, he_polar, core)

        return {"core": core, "ok": True, "seconds": round(time.time() - t0, 1),
                **{k: v for k, v in res.__dict__.items() if k != "core"}}
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config.ensure_project_dirs()
    circles = load_circles()

    srs = {p.name[: -len(config.SRS_SUFFIX)]: p
           for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}
    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}
    cores = config.usable_cores(
        sorted(set(srs) & set(he) & set(circles), key=core_sort_key))
    if args.limit:
        cores = cores[: args.limit]

    print(f"orientation test on {len(cores)} cores "
          f"(excluded: {sorted(config.EXCLUDED_CORES)})")

    jobs = [(c, srs[c], he[c], circles[c]) for c in cores]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_core, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"  [{i:3d}/{len(jobs)}] {r['core']:6s} FAILED {r['error']}")
    print(f"total {time.time() - t0:.0f}s")

    results.sort(key=lambda r: core_sort_key(r["core"]))
    good = [r for r in results if r["ok"]]
    if not good:
        print("no cores succeeded")
        return 1

    with (config.RESULTS_DIR / "04_orientation.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(good[0].keys()))
        w.writeheader()
        w.writerows(good)

    n_mirror = sum(r["best_mirrored"] for r in good)
    n = len(good)
    margins = np.array([r["margin"] for r in good])

    print("\n" + "=" * 74)
    print("MIRROR PARITY")
    print("=" * 74)
    print(f"  cores tested            : {n}")
    print(f"  prefer MIRRORED H&E     : {n_mirror}  ({n_mirror/n*100:.1f}%)")
    print(f"  prefer DIRECT (no flip) : {n - n_mirror}  ({(n-n_mirror)/n*100:.1f}%)")
    print(f"\n  margin (mirrored - direct), positive => mirror is better:")
    print(f"    median {np.median(margins):+.5f}   mean {margins.mean():+.5f}   "
          f"sd {margins.std(ddof=1):.5f}")
    print(f"    range  [{margins.min():+.5f}, {margins.max():+.5f}]")

    # Sign test: under the null that parity is a coin flip per core, the count
    # of mirror-preferring cores is Binomial(n, 0.5).
    from math import comb
    k = max(n_mirror, n - n_mirror)
    p_two_sided = 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    print(f"\n  sign test vs 50/50: p = {min(p_two_sided, 1.0):.3g}")

    dissenters = [r["core"] for r in good
                  if r["best_mirrored"] != (n_mirror > n / 2)]
    print(f"  cores dissenting from the majority: {len(dissenters)} {dissenters}")

    angles = np.array([r["best_angle_deg"] for r in good])
    print("\n" + "=" * 74)
    print("ROTATION (H&E -> SRS, under the winning parity)")
    print("=" * 74)
    hist, edges = np.histogram(angles, bins=12, range=(0, 360))
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"    {e0:5.0f}-{e1:5.0f} deg  {'#' * int(h)} {h}")
    # Circular concentration: |mean resultant vector|. Near 0 => scattered.
    rad = np.deg2rad(angles)
    Rbar = float(np.hypot(np.cos(rad).mean(), np.sin(rad).mean()))
    print(f"\n  circular concentration |R| = {Rbar:.3f} "
          f"(0 = uniform, 1 = all identical)")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    majority_mirror = n_mirror > n / 2
    # The verdict rests on the sign test, not on an arbitrary dissenter count.
    # An earlier version additionally required <=3 dissenters and so reported
    # "NOT consistent" at 70/75 with p = 9.8e-16, which was plainly wrong: a
    # handful of dissenters is expected whenever some cores carry weak
    # structural signal, and the sign test already accounts for them.
    if p_two_sided < 0.001:
        print(f"  A consistent mirror IS present: H&E is "
              f"{'MIRRORED' if majority_mirror else 'NOT mirrored'} "
              f"relative to SRS.")
        print(f"  {n - len(dissenters)}/{n} cores agree, p = {p_two_sided:.3g}.")
        if dissenters:
            print(f"  {len(dissenters)} dissenting core(s) to inspect:"
                  f" {dissenters}")
    else:
        print("  Parity is NOT consistent across cores. Either the structural"
              " signal is too weak, or there is no global mirror convention.")

    if Rbar > 0.5:
        print(f"  Rotations are concentrated (|R| = {Rbar:.3f}) - corroborating."
              " The cores share one slide in a fixed grid, loaded consistently"
              " on both instruments, so a shared rotation is expected.")
    else:
        print(f"  WARNING: rotations are near-uniform (|R| = {Rbar:.3f}). The"
              " correlation is not locking onto shared structure; treat the"
              " parity result with suspicion.")

    (config.RESULTS_DIR / "04_orientation.json").write_text(json.dumps(
        {"n": n, "n_mirrored": n_mirror, "p_sign_test": p_two_sided,
         "dissenters": dissenters, "circular_concentration": Rbar,
         "results": good}, indent=2), encoding="utf-8")
    print(f"\n  wrote {config.RESULTS_DIR / '04_orientation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
