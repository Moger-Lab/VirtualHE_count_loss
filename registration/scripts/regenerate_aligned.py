"""Regenerate the v2 aligned stacks from the raw source data.

STANDALONE. Depends only on numpy, tifffile, Pillow and scipy - no code from
the VirtualH&E_Analysis_v2 project. Copy this next to `16_transforms.csv` and
`checksums.csv` and run it.

What it produces, per core, in --out:

    <core>_srs.npy   (4, 4408, 4408) uint16   SRS, native, unresampled
    <core>_he.npy    (4408, 4408, 3) uint8    H&E warped onto the SRS grid

Both are indexed by the SAME (y0, x0, size) from `15_patch_manifest.csv`.

`_valid.npy` is deliberately NOT regenerated. Validity is already applied in
the patch manifest - every row there is >= 85% valid - so the mask is only
needed if you want to re-cut patches with a different scheme.

Usage
-----
    python regenerate_aligned.py ^
        --data "<folder containing the raw fsSRS and H&E images>" ^
        --transforms 16_transforms.csv ^
        --out <output folder> ^
        --workers 5

Add --verify to check every output against `checksums.csv`. Expect ~2 minutes
with 5 workers, and ~15 GB of output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

SRS_SUFFIX = "_fSRS.tif"
HE_OPTIONAL_SUFFIX = "_H&E"


def he_core_id(path: Path) -> str:
    """H&E filenames follow TWO conventions in this cohort.

    Row A is bare `<core>.png`; rows B-H carry `<core>_H&E.png`. Parsing on
    either alone silently drops the other set with no error.
    """
    stem = path.stem
    if stem.endswith(HE_OPTIONAL_SUFFIX):
        stem = stem[: -len(HE_OPTIONAL_SUFFIX)]
    return stem


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def blake(path: Path) -> str:
    a = np.load(path, mmap_mode="r")
    h = hashlib.blake2b(digest_size=16)
    step = max(1, a.shape[0] // 32)
    for i in range(0, a.shape[0], step):
        h.update(np.ascontiguousarray(a[i:i + step]).tobytes())
    return h.hexdigest()


def build(args) -> dict:
    core, srs_path, he_path, row, outdir, verify, expect = args
    t0 = time.time()
    try:
        outdir = Path(outdir)

        vol = tifffile.imread(srs_path)
        if vol.ndim != 3:
            raise ValueError(f"unexpected SRS shape {vol.shape}")
        if vol.shape[0] != 4 and vol.shape[-1] == 4:
            vol = np.moveaxis(vol, -1, 0)
        H, W = vol.shape[1], vol.shape[2]

        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))

        scale = float(row["scale_he_px_per_srs_px"])
        mirror = bool(int(row["mirror_he_lr"]))
        srs_cy = float(row["srs_circle_cy"]) + float(row["refine_dy"])
        srs_cx = float(row["srs_circle_cx"]) + float(row["refine_dx"])
        he_cy = float(row["he_circle_cy"])
        he_cx = float(row["he_circle_cx"])
        he_w = int(row["he_width"])

        # Mirror first; the fitted centre moves with the array.
        src = rgb[:, ::-1] if mirror else rgb
        he_cx_m = (he_w - 1 - he_cx) if mirror else he_cx

        ys_s, xs_s = np.mgrid[0:H, 0:W].astype(np.float32)
        ys = he_cy + (ys_s - srs_cy) * scale
        xs = he_cx_m + (xs_s - srs_cx) * scale
        del ys_s, xs_s

        # cval=255 -> white, matching slide background, so optical-density
        # maths sees background rather than a spurious dark border.
        chans = [ndi.map_coordinates(src[..., c].astype(np.float32), [ys, xs],
                                     order=1, mode="constant", cval=255.0)
                 for c in range(3)]
        he_w_arr = np.clip(np.stack(chans, axis=-1), 0, 255).astype(np.uint8)

        p_srs = outdir / f"{core}_srs.npy"
        p_he = outdir / f"{core}_he.npy"
        np.save(p_srs, vol)
        np.save(p_he, he_w_arr)

        out = {"core": core, "ok": True, "seconds": round(time.time() - t0, 1)}
        if verify and expect:
            # SRS is a verbatim TIFF repackaging - must be bit-identical.
            want = expect.get("srs_blake2b", "")
            out["srs_match"] = (blake(p_srs) == want) if want else None

            # H&E is a floating-point warp. Two correct implementations differ
            # by ~1 grey level through rounding, and more across numpy/scipy
            # versions, so an exact hash would fail on correct output. Compare
            # summary statistics with a tolerance instead.
            try:
                m = float(expect.get("he_mean", "nan"))
                s = float(expect.get("he_std", "nan"))
                got_m, got_s = float(he_w_arr.mean()), float(he_w_arr.std())
                out["he_dmean"] = round(got_m - m, 4)
                out["he_dstd"] = round(got_s - s, 4)
                out["he_match"] = bool(abs(got_m - m) < 0.05
                                       and abs(got_s - s) < 0.05
                                       and expect.get("he_shape", "")
                                       == "x".join(str(v)
                                                   for v in he_w_arr.shape))
            except (TypeError, ValueError):
                out["he_match"] = None
        return out
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="folder with the raw paired images")
    ap.add_argument("--transforms", default="16_transforms.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--verify", action="store_true",
                    help="check outputs against checksums.csv")
    ap.add_argument("--checksums", default="checksums.csv")
    args = ap.parse_args()

    data = Path(args.data)
    srs_dir, he_dir = data / "fsSRS", data / "H&E"
    for d in (srs_dir, he_dir):
        if not d.is_dir():
            print(f"ERROR: not found: {d}")
            return 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = {r["core"]: r for r in csv.DictReader(
        Path(args.transforms).open(encoding="utf-8-sig"))}
    expect = {}
    if args.verify:
        cpath = Path(args.checksums)
        if cpath.exists():
            expect = {r["core"]: r for r in csv.DictReader(
                cpath.open(encoding="utf-8-sig"))}
        else:
            print(f"WARNING: {cpath} not found, skipping verification")

    srs = {p.name[: -len(SRS_SUFFIX)]: p for p in srs_dir.glob(f"*{SRS_SUFFIX}")}
    he = {he_core_id(p): p for p in he_dir.glob("*.png")}

    cores = sorted(set(rows) & set(srs) & set(he), key=core_sort_key)
    missing = sorted(set(rows) - set(cores), key=core_sort_key)
    print(f"regenerating {len(cores)} cores -> {outdir}")
    if missing:
        print(f"  WARNING: no source for {missing}")

    jobs = [(c, srs[c], he[c], rows[c], str(outdir), args.verify,
             expect.get(c)) for c in cores]

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"  {r['core']:6s} FAILED {r['error']}")
            elif i % 15 == 0:
                print(f"  [{i:3d}/{len(jobs)}] ...")
    print(f"total {time.time() - t0:.0f}s")

    good = [r for r in results if r["ok"]]
    print(f"\n  built {len(good)}/{len(jobs)}")
    if args.verify and expect:
        bad_srs = [r for r in good if r.get("srs_match") is False]
        bad_he = [r for r in good if r.get("he_match") is False]
        n_srs = len([r for r in good if r.get("srs_match") is not None])
        n_he = len([r for r in good if r.get("he_match") is not None])
        print(f"  SRS bit-identical : {n_srs - len(bad_srs)}/{n_srs}")
        print(f"  H&E within tolerance: {n_he - len(bad_he)}/{n_he}")
        dm = [abs(r["he_dmean"]) for r in good if "he_dmean" in r]
        if dm:
            print(f"    max |mean difference| {max(dm):.4f} grey levels "
                  f"(tolerance 0.05)")
        if bad_srs:
            print(f"  SRS MISMATCHES: {[r['core'] for r in bad_srs]}")
        if bad_he:
            print(f"  H&E MISMATCHES: {[r['core'] for r in bad_he]}")
        if bad_srs or bad_he:
            return 1
        print("  verification passed.")

    size = sum(f.stat().st_size for f in outdir.glob("*.npy")) / 1024 ** 3
    print(f"  on disk: {size:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

