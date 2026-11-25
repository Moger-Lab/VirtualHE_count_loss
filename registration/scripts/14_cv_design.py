"""Stage 5b - cross-validation design.

HANDOVER.md section 6 leaves the CV scheme to this work rather than inheriting
v1's arbitrary 48/10/11, and recommends that v1's test cores remain recoverable
as a held-out group so that CV performance and a direct legacy comparison can
both be reported.

Design
------
**Grouped by core.** Patches from one core share tissue, staining and
registration, so splitting patches rather than cores would leak.

**Stratified on nuclear density and TMA row.** Density because the entire
project target is discriminating density, so every fold must span the range -
a fold accidentally short of dense cores would make the headline metric
unmeasurable on it. Row because stage 0 established that acquisition session
and TMA row are almost perfectly confounded in this cohort, so balancing row
balances session too, and neither can be balanced independently.

**v1's test cores form a fixed legacy group**, never used for training in any
fold. They are `B-8, C-2, C-3, C-4, E-7, F-4, F-6, F-8, G-1, G-4, H-6` - read
from v1's manifest as core identifiers only; no v1 code or patch geometry is
reused.

Note v1 used **no A-row cores at all** (its cohort was rows B-H). That is
independent corroboration of the A-row problems found here in stages 0-3, and
it means the legacy comparison is unaffected by how A-row cores are handled.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/14_cv_design.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config

N_FOLDS = 5
SEED = 20260922

# v1 test cores, from its patch_manifest.csv `split` column. Identifiers only.
V1_TEST = ("B-8", "C-2", "C-3", "C-4", "E-7", "F-4", "F-6", "F-8",
           "G-1", "G-4", "H-6")
V1_VAL = ("B-1", "C-10", "C-9", "D-2", "E-6", "E-9", "F-2", "F-7", "G-3", "G-9")


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def row_of(core: str) -> str:
    return core.split("-")[0]


def main() -> int:
    config.ensure_project_dirs()
    tiers = json.loads(
        (config.RESULTS_DIR / "10_cohort_tiers.json").read_text(encoding="utf-8"))
    dens = {r["core"]: float(r["median"]) for r in csv.DictReader(
        (config.RESULTS_DIR / "09_nuclear_density.csv").open(encoding="utf-8"))}

    train_pool = set(tiers["train"])
    eval_ok = set(tiers["eval_eligible"])
    unpaired = set(tiers["unpaired_only"])

    legacy = sorted(set(V1_TEST) & train_pool, key=core_sort_key)
    legacy_dropped = sorted(set(V1_TEST) - train_pool, key=core_sort_key)
    rotating = sorted(train_pool - set(legacy), key=core_sort_key)

    print("=" * 74)
    print("CV DESIGN")
    print("=" * 74)
    print(f"  train pool          : {len(train_pool)} cores")
    print(f"  legacy group (v1 test): {len(legacy)} {legacy}")
    if legacy_dropped:
        print(f"    v1 test cores not in our pool: {legacy_dropped}")
    print(f"  rotating through folds: {len(rotating)} cores")
    print(f"  unpaired-only (never evaluated): {len(unpaired)} {sorted(unpaired)}")

    # --- stratified grouped assignment -------------------------------------
    # Serpentine dealing over a GLOBALLY density-sorted list.
    #
    # An earlier version sorted within row and dealt round-robin, restarting
    # the cursor per row. That balanced rows but not density: fold minima came
    # out at 10-26 and maxima at 74-112, so some folds held no sparse cores and
    # others no dense ones. Since the headline metric is density
    # discrimination, a fold missing an end of the range cannot measure it.
    #
    # Density therefore leads. Sorting globally and dealing 0,1,2,3,4 then
    # 4,3,2,1,0 alternately gives each fold one core from each density stratum,
    # balancing both the median and the range - plain round-robin balances the
    # median but lets the extremes accumulate in whichever fold the cycle
    # happens to land on. Row balance is checked afterwards as a secondary
    # property rather than imposed.
    folds: dict[int, list[str]] = {k: [] for k in range(N_FOLDS)}
    ordered = sorted(rotating, key=lambda c: (dens.get(c, 0.0), c))
    for i, c in enumerate(ordered):
        block, pos = divmod(i, N_FOLDS)
        k = pos if block % 2 == 0 else (N_FOLDS - 1 - pos)
        folds[k].append(c)

    print("\n" + "=" * 74)
    print(f"{N_FOLDS}-FOLD COMPOSITION")
    print("=" * 74)
    for k in range(N_FOLDS):
        cs = sorted(folds[k], key=core_sort_key)
        d = np.array([dens[c] for c in cs if c in dens])
        rows = Counter(row_of(c) for c in cs)
        n_eval = sum(1 for c in cs if c in eval_ok)
        print(f"\n  fold {k}: {len(cs)} cores ({n_eval} eval-eligible)")
        print(f"    density median {np.median(d):5.1f}  "
              f"range [{d.min():.0f}, {d.max():.0f}]  "
              f"IQR [{np.percentile(d,25):.0f}, {np.percentile(d,75):.0f}]")
        print(f"    rows {dict(sorted(rows.items()))}")
        print(f"    {cs}")

    # --- balance check ------------------------------------------------------
    print("\n" + "=" * 74)
    print("BALANCE")
    print("=" * 74)
    meds = [float(np.median([dens[c] for c in folds[k] if c in dens]))
            for k in range(N_FOLDS)]
    mins = [float(np.min([dens[c] for c in folds[k] if c in dens]))
            for k in range(N_FOLDS)]
    maxs = [float(np.max([dens[c] for c in folds[k] if c in dens]))
            for k in range(N_FOLDS)]
    print(f"  fold density medians: {[round(m,1) for m in meds]}")
    print(f"    spread {max(meds)-min(meds):.1f} nuclei/patch "
          f"({(max(meds)-min(meds))/np.mean(meds)*100:.1f}% of the mean)")
    print(f"  fold minima : {[round(m,0) for m in mins]}")
    print(f"  fold maxima : {[round(m,0) for m in maxs]}")
    lo, hi = min(mins), max(maxs)
    covers = all(mn <= lo * 1.6 and mx >= hi * 0.6 for mn, mx in zip(mins, maxs))
    print(f"  every fold spans a comparable density range: {covers}")

    ld = np.array([dens[c] for c in legacy if c in dens])
    ad = np.array([dens[c] for c in rotating if c in dens])
    print(f"\n  legacy group density: median {np.median(ld):.1f}, "
          f"range [{ld.min():.0f}, {ld.max():.0f}]")
    print(f"  rotating pool density: median {np.median(ad):.1f}, "
          f"range [{ad.min():.0f}, {ad.max():.0f}]")

    out = {
        "n_folds": N_FOLDS, "seed": SEED,
        "legacy_group": legacy,
        "legacy_not_available": legacy_dropped,
        "unpaired_only": sorted(unpaired),
        "eval_eligible": sorted(eval_ok, key=core_sort_key),
        "folds": {str(k): sorted(folds[k], key=core_sort_key)
                  for k in range(N_FOLDS)},
        "v1_reference_splits": {"test": list(V1_TEST), "val": list(V1_VAL)},
    }
    (config.RESULTS_DIR / "14_cv_design.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    with (config.RESULTS_DIR / "14_cv_design.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["core", "group", "fold", "density_median",
                    "eval_eligible", "row"])
        for c in sorted(train_pool | unpaired, key=core_sort_key):
            if c in unpaired:
                grp, fold = "unpaired_only", ""
            elif c in legacy:
                grp, fold = "legacy_test", ""
            else:
                grp = "rotating"
                fold = next(k for k in range(N_FOLDS) if c in folds[k])
            w.writerow([c, grp, fold, round(dens.get(c, float("nan")), 1),
                        c in eval_ok, row_of(c)])
    print(f"\n  wrote {config.RESULTS_DIR / '14_cv_design.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
