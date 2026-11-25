"""Stage 3 - register H&E to SRS, then validate it as a hard gate.

HANDOVER.md section 3 step 4: "REGISTRATION VALIDATION - a hard gate. Do not
proceed to patch extraction until you have demonstrated, with evidence, that
the registration is as good as parallel sections permit. Note that validating
at pixel level is impossible by construction, so you must define and justify
the scale at which you validate. Report per-core quality and exclude failures."

Why pixel-level validation is impossible
----------------------------------------
SRS and H&E are parallel serial sections. The nuclei in one are physically
different cells from those in the other, so pixel correspondence does not exist
and no registration method can create it. v1 measured how correspondence grows
with spatial scale: R^2 = 0.16 at 6 um, 0.28 at 12.5, 0.40 at 25, 0.48 at 50,
0.53 at 100. Any validation must therefore be stated at a scale, and that scale
must be justified rather than assumed.

What is measured here
---------------------
1. **Residual translation.** After applying the transform built from stage 1-2
   parameters, the cross-correlation peak between the two tissue-density maps
   should sit at zero offset. A large residual means the circle-centre
   translation is wrong for that core. The refinement is then applied and the
   residual recorded as a per-core quality number.

2. **Correspondence versus scale.** Correlation is recomputed after block-
   averaging to 6, 12.5, 25, 50 and 100 um. This reproduces v1's curve on an
   independent registration and shows the scale at which correspondence becomes
   usable.

3. **A null from mismatched cores.** Every core's SRS is correlated against
   every *other* core's registered H&E. Both are tissue-filled discs, so some
   correlation arises from disc overlap alone; the null quantifies exactly that
   baseline. A matched score that does not clear the null is not evidence of
   registration, however high it looks in absolute terms.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/06_registration.py
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
from skimage.registration import phase_cross_correlation
from skimage.transform import resize

import config
from src import coreseg, register

Image.MAX_IMAGE_PIXELS = None

DS = 4           # 4408/4 = 1102 px; ~0.91 um per px at the derived pixel size
CANON = 512      # common grid so any two cores are comparable
SRS_UM_PER_PX = config.SRS_UM_PER_PX
SCALES_UM = (6.0, 12.5, 25.0, 50.0, 100.0)
# H&E native px per SRS native px, pooled over clean cores (stage 1).
# 1.02403 +/- 0.00104 (SEM, n=56) after the crop-aware circle fit; the
# pre-crop-fix estimate was 1.02459 +/- 0.00107 (n=52). The two agree to well
# within one SEM, which is itself reassuring: excluding crop-line chords added
# four cores to the clean pool without shifting the constant.
COHORT_SCALE = config.COHORT_SCALE


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def load_stage_results():
    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row
    orient: dict[str, dict] = {}
    with (config.RESULTS_DIR / "04_orientation.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            orient[row["core"]] = row
    return shapes, orient


def cohort_mirror(orient: dict[str, dict]) -> bool:
    """The single mirror parity that applies to every core.

    A mirror arises from the optical path and slide-handling convention - one
    microscope, one scanner, one slide. It is a property of the *setup*, not of
    an individual core, so it cannot legitimately vary within the cohort.

    Stage 2 measured parity per core in order to *test* that claim, and the
    test passed overwhelmingly (70/75, p = 9.8e-16). The five dissenters are
    cores where the structural signal was too weak to call parity, not cores
    with genuinely different optics.

    An earlier version of this script used each core's own measured parity.
    That was wrong in principle and wrong in effect: it applied an inverted
    mirror to exactly those five cores, which then presented as large residual
    translations and negative correlations, and were duly mis-reported as
    registration failures.
    """
    votes = sum(r["best_mirrored"] == "True" for r in orient.values())
    return votes > len(orient) / 2


def canonicalise(img: np.ndarray, cy: float, cx: float, r: float,
                 size: int = CANON) -> np.ndarray:
    """Resample the fitted disc onto a fixed square grid."""
    g = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(g, g)
    return ndi.map_coordinates(img.astype(np.float32),
                               [cy + Y * r, cx + X * r],
                               order=1, mode="constant", cval=np.nan)


def disc_mask(size: int = CANON, frac: float = 0.92) -> np.ndarray:
    g = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(g, g)
    return (X ** 2 + Y ** 2) <= frac ** 2


def zscore_in_mask(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = a[m & np.isfinite(a)]
    if v.size < 16 or v.std() < 1e-8:
        return np.zeros_like(a)
    return np.where(np.isfinite(a), (a - v.mean()) / v.std(), 0.0)


def process_core(args) -> dict:
    core, srs_path, he_path, shp, ori, mirror = args
    t0 = time.time()
    try:
        s, h = shp["srs"], shp["he"]
        own_parity = ori["best_mirrored"] == "True"

        # --- tissue-density maps, both at 1/DS -----------------------------
        vol = tifffile.imread(srs_path)
        cars = vol[coreseg.TISSUE_CHANNELS[0]].astype(np.float32)
        del vol
        sy, sx = cars.shape
        srs_d = resize(cars, (sy // DS, sx // DS), order=1,
                       anti_aliasing=True, preserve_range=True)
        lo, hi = np.percentile(srs_d, (1.0, 99.5))
        srs_d = np.clip((srs_d - lo) / max(hi - lo, 1e-6), 0, 1)
        srs_d = filters.gaussian(srs_d, sigma=1.5)

        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        hy, hx = rgb.shape[:2]
        he_small = resize(rgb.astype(np.float32), (hy // DS, hx // DS, 3),
                          order=1, anti_aliasing=True, preserve_range=True)
        del rgb
        he_od = -np.log10(np.clip(he_small, 1.0, 255.0) / 255.0).sum(axis=-1)
        he_od = filters.gaussian(he_od, sigma=1.5)

        # --- transform, with all parameters expressed at 1/DS --------------
        t = register.Transform(
            core=core, scale=COHORT_SCALE, mirror=mirror,
            srs_cy=float(s["circle_cy_full"]) / DS,
            srs_cx=float(s["circle_cx_full"]) / DS,
            he_cy=float(h["circle_cy_full"]) / DS,
            he_cx=float(h["circle_cx_full"]) / DS,
            he_width=he_small.shape[1],
        )
        out_shape = srs_d.shape
        he_w = register.warp_gray_to_srs(he_od, t, out_shape, cval=np.nan)

        # --- residual translation ------------------------------------------
        srs_r = float(s["circle_diam_full"]) / 2.0 / DS
        gy, gx = np.mgrid[0:out_shape[0], 0:out_shape[1]]
        inside = ((gy - t.srs_cy) ** 2 + (gx - t.srs_cx) ** 2) <= (0.92 * srs_r) ** 2
        taper = register.soft_disc(out_shape, t.srs_cy, t.srs_cx, srs_r)

        a = zscore_in_mask(srs_d, inside) * taper
        b = zscore_in_mask(np.nan_to_num(he_w), inside) * taper
        # Bound the search to 10% of the punch radius: stage 1 established the
        # circle centres to a few percent, so anything beyond this is noise.
        max_shift = max(4, int(round(0.10 * srs_r)))
        dy, dx, peak = register.refine_shift(a, b, max_shift)
        resid_px_native = float(np.hypot(dy, dx) * DS)
        resid_um = resid_px_native * SRS_UM_PER_PX

        # --- re-warp with the refinement applied ---------------------------
        t.dy, t.dx = -dy, -dx
        he_w2 = register.warp_gray_to_srs(he_od, t, out_shape, cval=np.nan)

        # --- canonical grids, for scale sweep and the null -----------------
        srs_c = canonicalise(srs_d, t.srs_cy + t.dy, t.srs_cx + t.dx, srs_r)
        he_c = canonicalise(np.nan_to_num(he_w2, nan=float(np.nanmedian(he_w2))),
                            t.srs_cy + t.dy, t.srs_cx + t.dx, srs_r)

        return {"core": core, "ok": True,
                "seconds": round(time.time() - t0, 1),
                "mirror": mirror,
                "own_parity": own_parity,
                "parity_dissent": bool(own_parity != mirror),
                "resid_dy_px": round(dy * DS, 2),
                "resid_dx_px": round(dx * DS, 2),
                "resid_px": round(resid_px_native, 2),
                "resid_um": round(resid_um, 2),
                "shift_peak": round(float(peak), 5),
                "max_shift_px": int(max_shift),
                "srs_canon": srs_c.astype(np.float32),
                "he_canon": he_c.astype(np.float32)}
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 1)}


def block_corr(a: np.ndarray, b: np.ndarray, m: np.ndarray, block: int
               ) -> float:
    """Pearson r between a and b after block-averaging to `block` px."""
    n = (a.shape[0] // block) * block
    if n < block * 2:
        return float("nan")
    def red(x):
        return x[:n, :n].reshape(n // block, block, n // block, block).mean((1, 3))
    ra, rb, rm = red(a), red(b), red(m.astype(np.float32))
    sel = (rm > 0.75) & np.isfinite(ra) & np.isfinite(rb)
    if sel.sum() < 8:
        return float("nan")
    x, y = ra[sel], rb[sel]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config.ensure_project_dirs()
    config.ensure_work_dirs()
    shapes, orient = load_stage_results()

    srs = {p.name[: -len(config.SRS_SUFFIX)]: p
           for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}
    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}
    cores = config.usable_cores(
        sorted(set(srs) & set(he) & set(shapes) & set(orient), key=core_sort_key))
    if args.limit:
        cores = cores[: args.limit]

    mirror = cohort_mirror({c: orient[c] for c in cores})
    dissent = [c for c in cores
               if (orient[c]["best_mirrored"] == "True") != mirror]

    print(f"registering {len(cores)} cores "
          f"(excluded: {sorted(config.EXCLUDED_CORES)})")
    print(f"  cohort scale {COHORT_SCALE}, DS={DS}, "
          f"{DS * SRS_UM_PER_PX:.3f} um per working px")
    print(f"  cohort mirror = {mirror} (applied to ALL cores - a property of "
          f"the optical setup, not of a core)")
    print(f"  cores whose own stage-2 parity dissented: {dissent}")

    jobs = [(c, srs[c], he[c], shapes[c], orient[c], mirror) for c in cores]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_core, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"  {r['core']:6s} FAILED {r['error']}")
    print(f"total {time.time() - t0:.0f}s")

    results.sort(key=lambda r: core_sort_key(r["core"]))
    good = [r for r in results if r["ok"]]
    if not good:
        print("no cores succeeded")
        return 1

    mask = disc_mask()
    canon = {r["core"]: (zscore_in_mask(r["srs_canon"], mask),
                         zscore_in_mask(r["he_canon"], mask)) for r in good}

    # --- 1. residual translation -------------------------------------------
    resid = np.array([r["resid_um"] for r in good])
    print("\n" + "=" * 74)
    print("1. RESIDUAL TRANSLATION after circle-centre alignment")
    print("=" * 74)
    print(f"  median {np.median(resid):6.2f} um   mean {resid.mean():6.2f}   "
          f"p90 {np.percentile(resid, 90):6.2f}   max {resid.max():6.2f}")
    worst = sorted(good, key=lambda r: -r["resid_um"])[:8]
    print("  largest residuals: " +
          ", ".join(f"{r['core']}={r['resid_um']:.1f}um" for r in worst))

    # --- 2. correspondence versus scale ------------------------------------
    print("\n" + "=" * 74)
    print("2. CORRESPONDENCE VERSUS SPATIAL SCALE (matched pairs)")
    print("=" * 74)
    um_per_canon_px = (2.0 * np.median(
        [float(shapes[c]["srs"]["circle_diam_full"]) for c in canon])
        / 2.0 * SRS_UM_PER_PX) / CANON
    print(f"  canonical grid: {CANON} px across the punch "
          f"= {um_per_canon_px:.3f} um per px")
    scale_rows = []
    for um in SCALES_UM:
        blk = max(1, int(round(um / um_per_canon_px)))
        rs = np.array([block_corr(a, b, mask, blk) for a, b in canon.values()])
        rs = rs[np.isfinite(rs)]
        scale_rows.append({"um": um, "block_px": blk, "n": int(rs.size),
                           "r_mean": float(rs.mean()),
                           "r_median": float(np.median(rs)),
                           "R2_mean": float((rs ** 2).mean())})
        print(f"    {um:6.1f} um (block {blk:3d} px)  "
              f"r mean {rs.mean():.3f}  median {np.median(rs):.3f}  "
              f"R^2 mean {(rs ** 2).mean():.3f}")
    print("\n  v1 reference (HANDOVER.md s2): R^2 = 0.16 @6um, 0.28 @12.5,"
          " 0.40 @25, 0.48 @50, 0.53 @100")

    # --- 3. null from mismatched cores -------------------------------------
    print("\n" + "=" * 74)
    print("3. NULL: matched vs MISMATCHED core pairs")
    print("=" * 74)
    rng = np.random.default_rng(0)
    names = list(canon)
    blk50 = max(1, int(round(50.0 / um_per_canon_px)))
    matched = np.array([block_corr(canon[c][0], canon[c][1], mask, blk50)
                        for c in names])
    null = []
    for _ in range(600):
        i, j = rng.choice(len(names), 2, replace=False)
        null.append(block_corr(canon[names[i]][0], canon[names[j]][1],
                               mask, blk50))
    null = np.array([v for v in null if np.isfinite(v)])
    matched = matched[np.isfinite(matched)]
    print(f"  at 50 um (block {blk50} px)")
    print(f"    matched   n={matched.size:3d}  mean {matched.mean():+.3f}  "
          f"sd {matched.std(ddof=1):.3f}  "
          f"range [{matched.min():+.3f}, {matched.max():+.3f}]")
    print(f"    MISmatched n={null.size:3d}  mean {null.mean():+.3f}  "
          f"sd {null.std(ddof=1):.3f}  "
          f"p95 {np.percentile(null, 95):+.3f}")
    sep = (matched.mean() - null.mean()) / null.std(ddof=1)
    print(f"    separation: {sep:.2f} null SD")
    n_pass = int((matched > np.percentile(null, 95)).sum())
    print(f"    cores above the null 95th percentile: {n_pass}/{matched.size}")

    # --- per-core table -----------------------------------------------------
    rows = []
    for r, c in zip(good, names):
        rm = block_corr(canon[c][0], canon[c][1], mask, blk50)
        rows.append({"core": c, "mirror": r["mirror"],
                     "resid_dy_px": r["resid_dy_px"],
                     "resid_dx_px": r["resid_dx_px"],
                     "resid_um": r["resid_um"],
                     "r_50um": round(float(rm), 4) if np.isfinite(rm) else "",
                     "passes_null95": bool(np.isfinite(rm) and
                                           rm > np.percentile(null, 95))})
    with (config.RESULTS_DIR / "06_registration.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    failures = [r["core"] for r in rows if not r["passes_null95"]]
    print("\n" + "=" * 74)
    print("GATE")
    print("=" * 74)
    print(f"  cores passing: {len(rows) - len(failures)}/{len(rows)}")
    print(f"  cores FAILING the null-95 criterion: {failures}")

    np.savez_compressed(config.CACHE_DIR / "canonical_maps.npz",
                        **{f"{c}_srs": canon[c][0] for c in canon},
                        **{f"{c}_he": canon[c][1] for c in canon})
    (config.RESULTS_DIR / "06_registration.json").write_text(json.dumps(
        {"cohort_scale": COHORT_SCALE, "ds": DS,
         "um_per_canon_px": um_per_canon_px,
         "scale_sweep": scale_rows,
         "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
         "null_p95": float(np.percentile(null, 95)),
         "matched_mean": float(matched.mean()),
         "separation_null_sd": float(sep),
         "failures": failures}, indent=2), encoding="utf-8")
    print(f"\n  wrote {config.RESULTS_DIR / '06_registration.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

