"""Stage 5d - export per-core registration parameters as one portable file.

Everything needed to reproduce the SRS<->H&E mapping without this project's
code, for consumers on other machines. The parameters currently live across
two result files (circle fits in stage 2, refinement in stage 6); this joins
them and states the transform explicitly.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/16_export_transforms.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import config

Image.MAX_IMAGE_PIXELS = None


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def main() -> int:
    config.ensure_project_dirs()

    shapes: dict[str, dict] = {}
    with (config.RESULTS_DIR / "02_core_shapes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            shapes.setdefault(row["core"], {})[row["modality"]] = row
    reg = {r["core"]: r for r in csv.DictReader(
        (config.RESULTS_DIR / "06_registration.csv").open(encoding="utf-8"))}
    tiers = {r["core"]: r for r in csv.DictReader(
        (config.RESULTS_DIR / "10_cohort_tiers.csv").open(encoding="utf-8"))}
    cvd = {r["core"]: r for r in csv.DictReader(
        (config.RESULTS_DIR / "14_cv_design.csv").open(encoding="utf-8"))}
    cov = {r["core"]: float(r["scan_coverage"]) for r in csv.DictReader(
        (config.RESULTS_DIR / "08_he_scan_coverage.csv").open(encoding="utf-8"))}

    he_files = {config.he_core_id(p): p
                for p in config.HE_DIR.glob(config.HE_GLOB)}

    rows = []
    for core in sorted(cvd, key=core_sort_key):
        if core not in shapes or core not in reg:
            continue
        s, h = shapes[core]["srs"], shapes[core]["he"]
        with Image.open(he_files[core]) as im:
            hw, hh = im.size          # PIL gives (width, height)
        rows.append({
            "core": core,
            "scale_he_px_per_srs_px": config.COHORT_SCALE,
            "mirror_he_lr": int(config.COHORT_MIRROR),
            "srs_circle_cy": repr(float(s["circle_cy_full"])),
            "srs_circle_cx": repr(float(s["circle_cx_full"])),
            "srs_circle_diam": round(float(s["circle_diam_full"]), 2),
            "he_circle_cy": repr(float(h["circle_cy_full"])),
            "he_circle_cx": repr(float(h["circle_cx_full"])),
            "he_circle_diam": round(float(h["circle_diam_full"]), 2),
            "he_width": hw, "he_height": hh,
            # Applied refinement, in SRS native px, ADDED to the SRS centre.
            "refine_dy": repr(-float(reg[core]["resid_dy_px"])),
            "refine_dx": repr(-float(reg[core]["resid_dx_px"])),
            "resid_um": reg[core]["resid_um"],
            "r_50um": reg[core]["r_50um"],
            "tier": tiers[core]["tier"],
            "scan_coverage": round(cov.get(core, 1.0), 4),
            "group": cvd[core]["group"],
            "fold": cvd[core]["fold"],
            "eval_eligible": cvd[core]["eval_eligible"],
        })

    out = config.RESULTS_DIR / "16_transforms.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "scale_he_px_per_srs_px": config.COHORT_SCALE,
        "scale_sem": 0.00098,
        "mirror_he_lr": config.COHORT_MIRROR,
        "rotation_deg": 0.0,
        "srs_um_per_px": config.SRS_UM_PER_PX,
        "he_um_per_px": config.HE_UM_PER_PX,
        "srs_shape": [4, 4408, 4408],
        "he_shape": [6000, 6000, 3],
        "srs_channels": config.SRS_CHANNELS,
        "excluded_cores": config.EXCLUDED_CORES,
        "stain_matrix": json.loads(
            (config.RESULTS_DIR / "11_stain_calibration.json").read_text(
                encoding="utf-8"))["H"],
        "n_cores": len(rows),
        "transform": (
            "he_y = he_cy + (srs_y - (srs_cy + refine_dy)) * scale; "
            "he_x = he_cx_mirrored + (srs_x - (srs_cx + refine_dx)) * scale; "
            "where he_cx_mirrored = (he_width - 1 - he_cx) if mirror else he_cx, "
            "and the H&E array is flipped left-right before sampling."),
    }
    (config.RESULTS_DIR / "16_transforms.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"exported {len(rows)} cores")
    print(f"  scale {config.COHORT_SCALE}, mirror {config.COHORT_MIRROR}, "
          f"rotation 0")
    from collections import Counter
    print(f"  tiers  {dict(Counter(r['tier'] for r in rows))}")
    print(f"  groups {dict(Counter(r['group'] for r in rows))}")
    print(f"  wrote {out}")
    print(f"  wrote {config.RESULTS_DIR / '16_transforms.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
