"""Stage 3b - how much of each punch is actually covered by H&E scan data?

Registration QC revealed several A-row H&E images cropped: a hard horizontal
edge with pure white above it, cutting off the top of the core. The tissue is
not faint there, it is absent - the scan does not cover that region.

Two earlier metrics both missed this, for instructive reasons:

  * stage 1 `border_frac` asks whether the thresholded tissue footprint touches
    the image edge. A cropped core's tissue stops at the crop line, which is in
    the middle of the frame, so it never touches an edge.
  * a punch-inside-frame check asks whether the fitted circle fits within
    6000x6000. It does - the missing region is blank canvas *inside* the frame,
    not outside it.

The question that actually matters is: what fraction of the fitted punch
contains scanned slide? Blank canvas is pure white (RGB ~255 with no texture);
real slide background is off-white and carries fine noise. That separation is
what this measures.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/08_he_scan_coverage.py
"""

from __future__ import annotations

import csv
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

Image.MAX_IMAGE_PIXELS = None

DS = 8
BLANK_LEVEL = 250.0     # RGB at/above this, with no local texture, is canvas
BLANK_TEXTURE = 0.6     # local std below this is "no texture"


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def blank_mask(small: np.ndarray) -> np.ndarray:
    """Pure-white, texture-free canvas - i.e. region the scanner did not cover."""
    from scipy import ndimage as ndi
    bright = small.min(axis=-1) >= BLANK_LEVEL
    g = small.mean(axis=-1)
    local_var = ndi.uniform_filter(g ** 2, 5) - ndi.uniform_filter(g, 5) ** 2
    flat = np.sqrt(np.clip(local_var, 0, None)) < BLANK_TEXTURE
    return bright & flat


def process(args) -> dict:
    core, he_path, cy, cx, r = args
    try:
        with Image.open(he_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        small = resize(rgb.astype(np.float32), (h // DS, w // DS, 3), order=1,
                       anti_aliasing=True, preserve_range=True)
        del rgb
        blank = blank_mask(small)

        gy, gx = np.mgrid[0:small.shape[0], 0:small.shape[1]]
        disc = ((gy - cy / DS) ** 2 + (gx - cx / DS) ** 2) <= (r / DS) ** 2
        if disc.sum() == 0:
            raise ValueError("empty disc")
        cov = 1.0 - float((blank & disc).sum()) / float(disc.sum())

        # A scan crop presents as a straight horizontal or vertical boundary.
        rows_blank = (blank & disc).sum(axis=1) / np.maximum(disc.sum(axis=1), 1)
        cols_blank = (blank & disc).sum(axis=0) / np.maximum(disc.sum(axis=0), 1)
        return {"core": core, "ok": True,
                "scan_coverage": round(cov, 4),
                "rows_fully_blank": int((rows_blank > 0.98).sum()),
                "cols_fully_blank": int((cols_blank > 0.98).sum())}
    except Exception as exc:  # noqa: BLE001
        return {"core": core, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    config.ensure_project_dirs()
    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row

    he = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}
    cores = config.usable_cores(
        sorted(set(he) & set(shapes), key=core_sort_key))

    jobs = []
    for c in cores:
        h = shapes[c]["he"]
        try:
            jobs.append((c, he[c], float(h["circle_cy_full"]),
                         float(h["circle_cx_full"]),
                         float(h["circle_diam_full"]) / 2.0))
        except (ValueError, TypeError):
            continue

    print(f"measuring H&E scan coverage for {len(jobs)} cores")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(process, j) for j in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())
    print(f"total {time.time() - t0:.0f}s")

    good = [r for r in results if r["ok"]]
    good.sort(key=lambda r: r["scan_coverage"])

    reg = {r["core"]: r for r in csv.DictReader(
        (config.RESULTS_DIR / "06_registration.csv").open(encoding="utf-8"))}

    print("\n" + "=" * 74)
    print("H&E SCAN COVERAGE OF THE FITTED PUNCH")
    print("=" * 74)
    print(f"{'core':7s}{'coverage':>10s}{'blank rows':>12s}{'resid_um':>10s}"
          f"{'r50':>8s}  pass")
    print("-" * 60)
    for r in good:
        if r["scan_coverage"] < 0.995:
            m = reg.get(r["core"], {})
            print(f"{r['core']:7s}{r['scan_coverage']:10.3f}"
                  f"{r['rows_fully_blank']:12d}"
                  f"{float(m.get('resid_um', 'nan')):10.1f}"
                  f"{float(m['r_50um']) if m.get('r_50um') else float('nan'):8.3f}"
                  f"  {m.get('passes_null95', '?')}")

    cov = np.array([r["scan_coverage"] for r in good])
    cropped = [r for r in good if r["scan_coverage"] < 0.995]
    complete = [r for r in good if r["scan_coverage"] >= 0.995]
    print(f"\n  fully scanned (>=99.5%) : {len(complete)}/{len(good)}")
    print(f"  cropped                  : {len(cropped)} "
          f"{[r['core'] for r in cropped]}")
    print(f"  coverage: median {np.median(cov):.4f}  min {cov.min():.4f}")

    def stat(rs, key):
        v = [float(reg[r["core"]][key]) for r in rs
             if r["core"] in reg and reg[r["core"]][key]]
        return np.median(v) if v else float("nan")

    if cropped:
        print(f"\n  cropped  : resid median {stat(cropped,'resid_um'):5.1f} um   "
              f"r50 median {stat(cropped,'r_50um'):+.3f}   "
              f"pass {sum(reg[r['core']]['passes_null95']=='True' for r in cropped if r['core'] in reg)}"
              f"/{len(cropped)}")
        print(f"  complete : resid median {stat(complete,'resid_um'):5.1f} um   "
              f"r50 median {stat(complete,'r_50um'):+.3f}   "
              f"pass {sum(reg[r['core']]['passes_null95']=='True' for r in complete if r['core'] in reg)}"
              f"/{len(complete)}")

    with (config.RESULTS_DIR / "08_he_scan_coverage.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(good[0].keys()))
        w.writeheader()
        w.writerows(sorted(good, key=lambda r: core_sort_key(r["core"])))
    print(f"\n  wrote {config.RESULTS_DIR / '08_he_scan_coverage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
