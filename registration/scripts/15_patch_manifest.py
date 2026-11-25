"""Stage 5c - patch manifest.

Coordinates only. The pixels stay in the memory-mapped per-core stacks written
by stage 5a, so a patch scheme is changed by rewriting a CSV rather than by
regenerating tens of GB.

Geometry
--------
100 um patches rendered at 512 px, matching v1 and the evaluation protocol
(HANDOVER.md section 6) so the finale is a like-for-like comparison. At
0.22701 um per SRS pixel that is 441 native px, so 512 is a mild upsample -
the brief notes the same of v1, and it is deliberate: StarDist was trained near
this scale.

Training patches overlap by 50%; evaluation patches do not overlap. Overlap is
augmentation for training but would put correlated patches on both sides of a
metric, so evaluation uses a disjoint tiling.

Every patch must be >= MIN_VALID of valid area - tissue present in BOTH
modalities and inside the scanned region.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/15_patch_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config

PATCH_UM = 100.0
RENDER_PX = 512
MIN_VALID = 0.85
TRAIN_OVERLAP = 0.5


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def tile(valid: np.ndarray, size: int, stride: int, min_valid: float):
    """Integral-image tiling: every window's valid fraction in one pass."""
    ii = np.zeros((valid.shape[0] + 1, valid.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(valid.astype(np.int64), axis=0), axis=1)
    out = []
    area = size * size
    for y0 in range(0, valid.shape[0] - size + 1, stride):
        for x0 in range(0, valid.shape[1] - size + 1, stride):
            y1, x1 = y0 + size, x0 + size
            s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            f = s / area
            if f >= min_valid:
                out.append((y0, x0, round(float(f), 4)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-um", type=float, default=PATCH_UM)
    args = ap.parse_args()

    config.ensure_project_dirs()
    aligned = config.WORK_ROOT / "aligned"

    cv = json.loads(
        (config.RESULTS_DIR / "14_cv_design.json").read_text(encoding="utf-8"))
    group_of, fold_of = {}, {}
    with (config.RESULTS_DIR / "14_cv_design.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            group_of[r["core"]] = r["group"]
            fold_of[r["core"]] = r["fold"]
    eval_ok = set(cv["eval_eligible"])

    size = int(round(args.patch_um / config.SRS_UM_PER_PX))
    stride_train = max(1, int(round(size * (1.0 - TRAIN_OVERLAP))))
    print(f"patch {args.patch_um:.0f} um = {size} SRS px, rendered at {RENDER_PX}")
    print(f"  train stride {stride_train} ({TRAIN_OVERLAP:.0%} overlap), "
          f"eval stride {size} (disjoint), min valid {MIN_VALID:.0%}")

    cores = sorted(group_of, key=core_sort_key)
    rows = []
    t0 = time.time()
    for i, core in enumerate(cores, 1):
        vp = aligned / f"{core}_valid.npy"
        if not vp.exists():
            print(f"  {core}: no aligned stack, skipped")
            continue
        valid = np.load(vp, mmap_mode="r")
        valid = np.asarray(valid)

        grp = group_of[core]
        for y0, x0, f in tile(valid, size, stride_train, MIN_VALID):
            rows.append({"core": core, "group": grp, "fold": fold_of[core],
                         "role": "train", "y0": y0, "x0": x0, "size": size,
                         "render": RENDER_PX, "valid_frac": f})
        if core in eval_ok:
            for y0, x0, f in tile(valid, size, size, MIN_VALID):
                rows.append({"core": core, "group": grp, "fold": fold_of[core],
                             "role": "eval", "y0": y0, "x0": x0, "size": size,
                             "render": RENDER_PX, "valid_frac": f})
        if i % 20 == 0:
            print(f"  [{i:3d}/{len(cores)}] ...")
    print(f"total {time.time() - t0:.0f}s")

    with (config.RESULTS_DIR / "15_patch_manifest.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 74)
    print("MANIFEST")
    print("=" * 74)
    tr = [r for r in rows if r["role"] == "train"]
    ev = [r for r in rows if r["role"] == "eval"]
    print(f"  train patches : {len(tr):6d}  over "
          f"{len({r['core'] for r in tr})} cores")
    print(f"  eval patches  : {len(ev):6d}  over "
          f"{len({r['core'] for r in ev})} cores")

    for grp in ("rotating", "legacy_test", "unpaired_only"):
        g = [r for r in tr if r["group"] == grp]
        if g:
            print(f"    {grp:15s} {len(g):6d} train patches, "
                  f"{len({r['core'] for r in g})} cores")

    per = [len([r for r in tr if r["core"] == c])
           for c in {r["core"] for r in tr}]
    print(f"\n  train patches per core: median {int(np.median(per))}, "
          f"range [{min(per)}, {max(per)}]")
    print(f"  v1 for scale: 1,047 patches over 9 cores (~116/core)")

    for k in sorted({r["fold"] for r in tr if r["fold"] != ""}):
        n = len([r for r in tr if r["fold"] == k])
        ne = len([r for r in ev if r["fold"] == k])
        print(f"    fold {k}: {n:6d} train / {ne:5d} eval patches")

    print(f"\n  wrote {config.RESULTS_DIR / '15_patch_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
