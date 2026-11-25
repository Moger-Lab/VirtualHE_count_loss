"""Stage 3 QC - render registered pairs from the cached canonical maps.

Reads `canonical_maps.npz` written by `06_registration.py`, so it is fast and
renders exactly the maps the gate scored - not a re-derivation that might
differ from them.

Run:
    & "$env:USERPROFILE\\miniforge3\\envs\\srs-analysis\\python.exe" \\
        scripts/07_registration_qc.py --cores A-5 A-9 A-10 D-3
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def core_sort_key(core: str):
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", nargs="+", required=True)
    ap.add_argument("--tag", default="reg")
    args = ap.parse_args()

    config.ensure_project_dirs()
    outdir = config.FIGURES_DIR / "06_registration"
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(config.CACHE_DIR / "canonical_maps.npz")
    meta = {r["core"]: r for r in csv.DictReader(
        (config.RESULTS_DIR / "06_registration.csv").open(encoding="utf-8"))}

    cores = sorted(args.cores, key=core_sort_key)
    fig, axes = plt.subplots(len(cores), 4, figsize=(17, 4.2 * len(cores)),
                             squeeze=False)

    for i, core in enumerate(cores):
        if f"{core}_srs" not in z:
            print(f"  skip {core}: not in cache")
            continue
        s = z[f"{core}_srs"]
        h = z[f"{core}_he"]
        m = meta.get(core, {})

        axes[i, 0].imshow(s, cmap="gray")
        axes[i, 0].set_title(f"{core}  SRS (CARS)", fontsize=10)

        # H&E optical density: tissue is HIGH, so invert for visual comparison
        # against SRS, where tissue is also high.
        axes[i, 1].imshow(h, cmap="gray")
        axes[i, 1].set_title("H&E registered (OD)", fontsize=10)

        def norm(a):
            v = a[np.isfinite(a)]
            if v.size == 0:
                return np.zeros_like(a)
            lo, hi = np.percentile(v, (2, 98))
            return np.clip((np.nan_to_num(a) - lo) / max(hi - lo, 1e-6), 0, 1)

        ov = np.zeros(s.shape + (3,), dtype=np.float32)
        ov[..., 1] = norm(s)
        ov[..., 0] = norm(h)
        ov[..., 2] = norm(h)
        axes[i, 2].imshow(ov)
        axes[i, 2].set_title("overlay  SRS green / H&E magenta", fontsize=10)

        # Difference of the two normalised maps: structure that appears in one
        # and not the other shows up as saturated colour.
        d = norm(s) - norm(h)
        axes[i, 3].imshow(d, cmap="coolwarm", vmin=-1, vmax=1)
        axes[i, 3].set_title(
            f"difference   resid {m.get('resid_um','?')}um   "
            f"r50 {m.get('r_50um','?')}   pass={m.get('passes_null95','?')}",
            fontsize=10)

        for a in axes[i]:
            a.axis("off")

    fig.tight_layout()
    out = outdir / f"{args.tag}.png"
    fig.savefig(out, dpi=78, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
