"""Final cohort tiering, and the train/eval split.

Tiering criterion - corrected
-----------------------------
An earlier version tiered on two things it had wrong:

1. **Scan coverage was treated as disqualifying.** It is not. A cropped scan
   limits how much *tissue* a core contributes; it does not make the alignment
   of the tissue that *is* present wrong. After the crop-aware circle fit,
   `A-2` is only 91% scanned yet reaches r = 0.651 at 50 um - better than the
   tier-A median. Coverage belongs on its own axis, as usable area.

2. **Residual translation was treated as final alignment quality.** It is not.
   The residual is the correction the refinement *applied*; the correlation is
   measured *after* that correction. A large residual means the circle-centre
   starting estimate was poor, not that the end result is.

So the primary criterion is the post-refinement correlation against the
mismatched-core null. Residual is retained as a secondary diagnostic, because a
residual at or near the search bound indicates the search may have saturated
and the correction cannot be trusted.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/10_cohort_tiers.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config

RESID_SUSPECT_UM = 25.0


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def main() -> int:
    config.ensure_project_dirs()
    reg = {r["core"]: r for r in csv.DictReader(
        (config.RESULTS_DIR / "06_registration.csv").open(encoding="utf-8"))}
    cov = {r["core"]: float(r["scan_coverage"]) for r in csv.DictReader(
        (config.RESULTS_DIR / "08_he_scan_coverage.csv").open(encoding="utf-8"))}
    meta = json.loads(
        (config.RESULTS_DIR / "06_registration.json").read_text(encoding="utf-8"))
    null_p95 = meta["null_p95"]

    rows = []
    for core, r in reg.items():
        r50 = float(r["r_50um"]) if r["r_50um"] else float("nan")
        resid = float(r["resid_um"])
        passes = np.isfinite(r50) and r50 > null_p95
        if passes:
            tier = "A_registered"
        elif resid < RESID_SUSPECT_UM:
            tier = "B_weak_signal"
        else:
            tier = "C_misaligned"
        rows.append({"core": core, "tier": tier, "r_50um": round(r50, 4),
                     "resid_um": resid, "scan_coverage": cov.get(core, 1.0),
                     "beats_null": passes})
    rows.sort(key=lambda r: core_sort_key(r["core"]))

    print("=" * 74)
    print(f"COHORT TIERS   (null 95th percentile = {null_p95:+.3f} at 50 um)")
    print("=" * 74)
    labels = {
        "A_registered": "registration DEMONSTRATED - beats the mismatched null",
        "B_weak_signal": "alignment sound, too little structure to demonstrate it",
        "C_misaligned": "fails the null AND needed a large correction",
    }
    for tier in ("A_registered", "B_weak_signal", "C_misaligned"):
        v = [r for r in rows if r["tier"] == tier]
        print(f"\n{tier}  n={len(v)}   {labels[tier]}")
        if v:
            print(f"   r50 median {np.median([r['r_50um'] for r in v]):+.3f}   "
                  f"resid median {np.median([r['resid_um'] for r in v]):5.1f} um   "
                  f"coverage median {np.median([r['scan_coverage'] for r in v]):.3f}")
            print(f"   {[r['core'] for r in v]}")

    print("\n" + "=" * 74)
    print("USABLE AREA (a separate axis - cropped scans contribute less tissue)")
    print("=" * 74)
    cropped = [r for r in rows if r["scan_coverage"] < 0.995]
    for r in sorted(cropped, key=lambda r: r["scan_coverage"]):
        print(f"   {r['core']:6s} coverage {r['scan_coverage']:.3f}  "
              f"r50 {r['r_50um']:+.3f}  resid {r['resid_um']:5.1f} um  "
              f"-> {r['tier']}")
    recovered = [r for r in cropped if r["tier"] == "A_registered"]
    print(f"\n   cropped cores now registering successfully: "
          f"{len(recovered)}/{len(cropped)} {[r['core'] for r in recovered]}")

    train = [r["core"] for r in rows if r["tier"] in ("A_registered",
                                                      "B_weak_signal")]
    # Evaluation additionally excludes the defective-stain cores. They train
    # fine - v1 kept all three in training with no ill effect - but a core
    # whose stain is wrong cannot serve as ground truth for a stain-fidelity
    # measurement. v1 had all three in training and none in test, which is why
    # its evaluation was never contaminated; the same applies here.
    evalset = [r["core"] for r in rows
               if r["tier"] == "A_registered"
               and r["scan_coverage"] >= 0.995
               and r["core"] not in config.V1_DEFECTIVE_HE_CORES]
    unpaired = [r["core"] for r in rows if r["tier"] == "C_misaligned"]

    print("\n" + "=" * 74)
    print("SPLIT")
    print("=" * 74)
    print(f"  TRAIN pool (quasi-paired losses active) : {len(train)} cores")
    print(f"  EVAL-eligible (tier A, full scan)       : {len(evalset)} cores")
    print(f"  UNPAIRED-only (adversarial+cycle only)  : {len(unpaired)} cores"
          f"  {unpaired}")
    print(f"  Excluded outright                       : "
          f"{len(config.EXCLUDED_CORES)} {sorted(config.EXCLUDED_CORES)}")

    with (config.RESULTS_DIR / "10_cohort_tiers.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (config.RESULTS_DIR / "10_cohort_tiers.json").write_text(json.dumps(
        {"null_p95": null_p95, "train": train, "eval_eligible": evalset,
         "unpaired_only": unpaired,
         "excluded": sorted(config.EXCLUDED_CORES)}, indent=2), encoding="utf-8")
    print(f"\n  wrote {config.RESULTS_DIR / '10_cohort_tiers.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
