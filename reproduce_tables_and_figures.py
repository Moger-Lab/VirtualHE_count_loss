"""
Level 1 reproduction: regenerate every statistic and the data figures of the
paper from the released per-patch tables (no images, no GPU; CPU, about
10-20 minutes), then compare each regenerated table with the released one.

    python reproduce_tables_and_figures.py --out <empty folder>

Regenerated: four-model table, per-run values and paired differences
(Fig. 3, Fig. 4, S4, S6a), descriptive by-class tables and checkpoint
fabrication (Fig. 5, S7), detector robustness (S5), registration quality
(S2), counter validation (S3), design rationale (S8; panel a uses the
released area-signal table, whose inference needs the image data).
Needs the images (data on request, see README): Figs 1, 2, S1, S6b, S9.
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(REPO, "count_loss", "paper_figures", "final_2026-10")
STEPS = ["a01_tables.py", "a02_descriptive.py", "a04_detector_paired.py", "fig3_four_models.py",
         "fig4_patch_level.py", "fig5_what_remains.py", "figS2_registration.py", "figS3_counter.py",
         "figS4_seeds_cores.py", "figS5_detectors.py", "figS7_tables.py", "figS8_design.py"]
# computed with image inference (a03) or image selection (g01): copied, not regenerated
PRECOMPUTED = ["area_signal_per_patch.csv"]


def compare(a, b):
    A, B = pd.read_csv(a), pd.read_csv(b)
    if list(A.columns) != list(B.columns) or len(A) != len(B):
        return "shape/columns differ"
    worst = 0.0
    for c in A.columns:
        if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c]):
            d = np.nanmax(np.abs(A[c].to_numpy(float) - B[c].to_numpy(float))) if len(A) else 0.0
            worst = max(worst, 0.0 if np.isnan(d) else d)
        elif not A[c].astype(str).equals(B[c].astype(str)):
            return f"column {c} differs"
    return "identical" if worst == 0 else f"max abs diff {worst:.2e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} is not empty")
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    for f in PRECOMPUTED:
        shutil.copy2(os.path.join(FIG, "data", f), os.path.join(out, "data", f))
    env = dict(os.environ, VHE_FIG_OUT=out, MPLBACKEND="Agg")
    for s in STEPS:
        print(f"running {s}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(FIG, s)], env=env, cwd=FIG, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-4000:])
            sys.exit(f"{s} failed")
    r = subprocess.run([sys.executable, "-c", "import figS6_cg_seeds as S, fcommon as F; F.apply(); S.seed_values()"],
                       env=env, cwd=FIG, capture_output=True, text=True)
    print("S6a:", "ok" if r.returncode == 0 else r.stderr[-2000:])

    print("\ncomparison with the released tables:")
    rows = []
    for f in sorted(os.listdir(os.path.join(out, "data"))):
        if f in PRECOMPUTED:
            continue
        ref = os.path.join(FIG, "data", f)
        rows.append((f, compare(os.path.join(out, "data", f), ref) if os.path.exists(ref) else "no released copy"))
    for f in sorted(os.listdir(out)):
        if f.endswith("_source.csv"):
            ref = os.path.join(FIG, f)
            rows.append((f, compare(os.path.join(out, f), ref) if os.path.exists(ref) else "no released copy"))
    for f, v in rows:
        print(f"  {f:60s} {v}")
    bad = [r for r in rows if not (r[1] == "identical" or r[1].startswith("max abs diff") and float(r[1].split()[-1]) < 1e-9)]
    print(f"\n{len(rows) - len(bad)} of {len(rows)} tables reproduced (identical or < 1e-9)")
    if bad:
        print("check:", bad)
