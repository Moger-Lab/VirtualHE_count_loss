"""Stage 5a - build per-core aligned stacks on the ML SSD.

For every usable core, write to `D:\\vhe_v2_work\\aligned`:

  <core>_srs.npy    (4, 4408, 4408) uint16  - SRS, NATIVE and unresampled
  <core>_he.npy     (4408, 4408, 3) uint8   - H&E warped onto the SRS grid
  <core>_valid.npy  (4408, 4408)    bool    - tissue present in BOTH modalities

Why store aligned whole cores rather than pre-cut patches
---------------------------------------------------------
A patch cache commits to a patch size, stride and render resolution before any
of those have been justified, and re-cutting means regenerating tens of GB.
Whole-core memory-mapped arrays let the manifest carry coordinates only, so a
patch scheme can be changed by rewriting a CSV. Slicing a memmap on the 1.8 GB/s
NVMe is fast enough to keep the GPU fed.

The SRS is stored **unresampled**. It is the model input, and interpolating it
once here would degrade every patch drawn from it forever. All geometric
correction is absorbed into the H&E side, which is the reference, not the input.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/13_build_aligned_stacks.py
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
from skimage.transform import resize

import config
from src import coreseg, register

Image.MAX_IMAGE_PIXELS = None

COHORT_SCALE = config.COHORT_SCALE
MASK_DS = 8          # masks are computed at 1/8 then upsampled - they are
                     # low-frequency objects and this is 64x cheaper


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def build(args) -> dict:
    core, srs_path, he_path, shp, reg_row, mirror, outdir = args
    t0 = time.time()
    try:
        outdir = Path(outdir)
        s, h = shp["srs"], shp["he"]

        vol = tifffile.imread(srs_path)
        if vol.shape[0] != 4 and vol.shape[-1] == 4:
            vol = np.moveaxis(vol, -1, 0)
        out_shape = vol.shape[1:]

        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))

        # Stage 3 measured the refinement in downsampled px and stored it in
        # NATIVE px as resid_d{y,x}_px, having APPLIED the negative of it.
        # Reproduce that sign convention exactly.
        t = register.Transform(
            core=core, scale=COHORT_SCALE, mirror=mirror,
            srs_cy=float(s["circle_cy_full"]), srs_cx=float(s["circle_cx_full"]),
            he_cy=float(h["circle_cy_full"]), he_cx=float(h["circle_cx_full"]),
            he_width=rgb.shape[1],
            dy=-float(reg_row["resid_dy_px"]), dx=-float(reg_row["resid_dx_px"]),
        )

        he_w = register.warp_he_to_srs(rgb, t, out_shape, order=1, cval=255.0)
        he_w = he_w.astype(np.uint8)

        # --- validity: tissue in BOTH modalities, and scanned in the H&E ---
        srs_small, _, _ = coreseg.segment_srs(vol, ds=MASK_DS)
        srs_foot = coreseg.footprint(srs_small)

        he_small = resize(he_w.astype(np.float32),
                          (out_shape[0] // MASK_DS, out_shape[1] // MASK_DS, 3),
                          order=1, anti_aliasing=True, preserve_range=True)
        he_raw, _, _, _ = coreseg.segment_he(he_w, ds=MASK_DS)
        he_foot = coreseg.footprint(he_raw)
        # Un-scanned canvas warps in as pure white; it is not tissue and must
        # never yield a patch.
        scanned = ~coreseg.blank_canvas(he_small)

        valid_small = srs_foot & he_foot & scanned
        valid_small = ndi.binary_erosion(valid_small, np.ones((3, 3)),
                                         iterations=2, border_value=0)
        valid = np.kron(valid_small, np.ones((MASK_DS, MASK_DS), dtype=bool))
        if valid.shape != out_shape:
            v = np.zeros(out_shape, dtype=bool)
            n0 = min(valid.shape[0], out_shape[0])
            n1 = min(valid.shape[1], out_shape[1])
            v[:n0, :n1] = valid[:n0, :n1]
            valid = v

        np.save(outdir / f"{core}_srs.npy", vol)
        np.save(outdir / f"{core}_he.npy", he_w)
        np.save(outdir / f"{core}_valid.npy", valid)

        return {"core": core, "ok": True, "seconds": round(time.time() - t0, 1),
                "valid_frac": round(float(valid.mean()), 4),
                "shape": list(out_shape)}
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    config.ensure_work_dirs()
    outdir = config.WORK_ROOT / "aligned"
    outdir.mkdir(parents=True, exist_ok=True)

    tiers = json.loads(
        (config.RESULTS_DIR / "10_cohort_tiers.json").read_text(encoding="utf-8"))
    cores = sorted(set(tiers["train"]) | set(tiers["unpaired_only"]),
                   key=core_sort_key)

    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row
    reg = {r["core"]: r for r in csv.DictReader(
        (config.RESULTS_DIR / "06_registration.csv").open(encoding="utf-8"))}

    mirror = True   # cohort convention, stage 2 (71/75, p = 6.8e-17)

    srs = {p.name[: -len(config.SRS_SUFFIX)]: p
           for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}
    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}

    jobs = [(c, srs[c], he[c], shapes[c], reg[c], mirror, str(outdir))
            for c in cores if c in srs and c in he and c in reg]

    print(f"building aligned stacks for {len(jobs)} cores -> {outdir}")
    print(f"  scale {COHORT_SCALE}, mirror {mirror}, "
          f"~{len(jobs) * 213 / 1024:.1f} GB expected")

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
    good.sort(key=lambda r: core_sort_key(r["core"]))
    vf = np.array([r["valid_frac"] for r in good])
    print(f"\n  built {len(good)}/{len(jobs)}")
    print(f"  valid fraction: median {np.median(vf):.3f}  "
          f"min {vf.min():.3f} ({good[int(vf.argmin())]['core']})  "
          f"max {vf.max():.3f}")

    size_gb = sum(f.stat().st_size for f in outdir.glob("*.npy")) / 1024 ** 3
    print(f"  on disk: {size_gb:.1f} GB")

    with (config.RESULTS_DIR / "13_aligned_stacks.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["core", "ok", "seconds",
                                           "valid_frac", "shape"])
        w.writeheader()
        for r in good:
            w.writerow({k: r.get(k, "") for k in
                        ["core", "ok", "seconds", "valid_frac", "shape"]})
    print(f"  wrote {config.RESULTS_DIR / '13_aligned_stacks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

