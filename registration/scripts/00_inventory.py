"""Stage 0 - data inventory.

Establishes, from the files themselves rather than from the brief:
  * which cores exist in each modality and which are paired
  * array shape / dtype / bit usage for both modalities
  * what the SRS TIFF metadata actually claims about pixel size
  * which files are un-hydrated OneDrive placeholders (must be downloaded
    before any bulk read, or reads will stall or fail)

Nothing here trusts HANDOVER.md; every claim it makes is re-checked and
disagreements are reported explicitly.

Run:
    & "$env:USERPROFILE\miniforge3\envs\srs-analysis\python.exe" scripts/00_inventory.py
"""

from __future__ import annotations

import csv
import json
import re
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tifffile
from PIL import Image

import config

Image.MAX_IMAGE_PIXELS = None  # 6000x6000 exceeds PIL's decompression-bomb guard

FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000


def is_cloud_placeholder(p: Path) -> bool:
    """True if the file's bytes are not resident on local disk."""
    attrs = getattr(p.stat(), "st_file_attributes", 0)
    return bool(attrs & (FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))


def core_sort_key(core: str):
    """Sort A-1, A-2, A-10 naturally rather than lexically."""
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def main() -> int:
    config.ensure_project_dirs()

    srs_files = {p.name[: -len(config.SRS_SUFFIX)]: p
                 for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}
    he_files = {config.he_core_id(p): p for p in config.HE_DIR.glob(config.HE_GLOB)}

    # The two H&E naming conventions, reported so the split is on record.
    he_naming = {"bare": [], "suffixed": []}
    for p in config.HE_DIR.glob(config.HE_GLOB):
        he_naming["suffixed" if p.stem.endswith(config.HE_OPTIONAL_SUFFIX)
                  else "bare"].append(config.he_core_id(p))

    srs_cores, he_cores = set(srs_files), set(he_files)
    paired = sorted(srs_cores & he_cores, key=core_sort_key)
    srs_only = sorted(srs_cores - he_cores, key=core_sort_key)
    he_only = sorted(he_cores - srs_cores, key=core_sort_key)

    print("=" * 74)
    print("COHORT")
    print("=" * 74)
    print(f"  SRS files      : {len(srs_cores)}")
    print(f"  H&E files      : {len(he_cores)}")
    print(f"  paired cores   : {len(paired)}")
    print(f"  SRS-only       : {srs_only}")
    print(f"  H&E-only       : {he_only}")
    print(f"  H&E naming     : {len(he_naming['bare'])} bare `<core>.png` "
          f"{sorted(he_naming['bare'], key=core_sort_key)}")
    print(f"                 : {len(he_naming['suffixed'])} suffixed `<core>_H&E.png`")

    # --- check the brief's claims ------------------------------------------
    disagreements: list[str] = []
    if len(paired) != config.EXPECTED_N_PAIRED:
        disagreements.append(
            f"paired count {len(paired)} != brief's {config.EXPECTED_N_PAIRED}")
    if tuple(srs_only) != config.EXPECTED_SRS_ONLY:
        disagreements.append(
            f"SRS-only {tuple(srs_only)} != brief's {config.EXPECTED_SRS_ONLY}")
    if tuple(he_only) != config.EXPECTED_HE_ONLY:
        disagreements.append(
            f"H&E-only {tuple(he_only)} != brief's {config.EXPECTED_HE_ONLY}")

    # --- hydration status ---------------------------------------------------
    srs_cloud = sorted((c for c, p in srs_files.items() if is_cloud_placeholder(p)),
                       key=core_sort_key)
    he_cloud = sorted((c for c, p in he_files.items() if is_cloud_placeholder(p)),
                      key=core_sort_key)
    print("\n" + "=" * 74)
    print("ONEDRIVE HYDRATION (un-downloaded files stall or fail on bulk read)")
    print("=" * 74)
    print(f"  SRS placeholders: {len(srs_cloud)} {srs_cloud if srs_cloud else ''}")
    print(f"  H&E placeholders: {len(he_cloud)} {he_cloud if he_cloud else ''}")

    # --- per-file geometry --------------------------------------------------
    print("\n" + "=" * 74)
    print("GEOMETRY (header reads, all paired cores)")
    print("=" * 74)

    rows = []
    srs_shapes, he_shapes = {}, {}
    for core in paired:
        row = {"core": core}
        with tifffile.TiffFile(srs_files[core]) as tf:
            series = tf.series[0]
            row["srs_shape"] = str(tuple(series.shape))
            row["srs_dtype"] = str(series.dtype)
            srs_shapes[str(tuple(series.shape))] = srs_shapes.get(str(tuple(series.shape)), 0) + 1
            page = tf.pages[0]
            tags = {t.name: t.value for t in page.tags}
            row["tif_xres"] = tags.get("XResolution")
            row["tif_resunit"] = str(tags.get("ResolutionUnit"))
            desc = tags.get("ImageDescription") or ""
            row["tif_desc"] = desc.replace("\n", " | ")[:300]
        with Image.open(he_files[core]) as im:
            row["he_shape"] = f"({im.size[1]}, {im.size[0]})"
            row["he_mode"] = im.mode
            he_shapes[(im.size, im.mode)] = he_shapes.get((im.size, im.mode), 0) + 1
        rows.append(row)

    print("  SRS shapes :", srs_shapes)
    print("  H&E shapes :", {f"{k[0]} {k[1]}": v for k, v in he_shapes.items()})

    # --- what the SRS metadata claims about scale ---------------------------
    print("\n" + "=" * 74)
    print("SRS TIFF METADATA (the brief says this is NOT authoritative)")
    print("=" * 74)
    sample = rows[0]
    print(f"  sample core     : {sample['core']}")
    print(f"  XResolution tag : {sample['tif_xres']}")
    print(f"  ResolutionUnit  : {sample['tif_resunit']}")
    print(f"  ImageDescription: {sample['tif_desc'][:400]}")
    descs = {r["tif_desc"] for r in rows}
    print(f"  distinct ImageDescription values across cohort: {len(descs)}")

    # --- bit usage, on a subsample -----------------------------------------
    print("\n" + "=" * 74)
    print("INTENSITY / BIT USAGE (6 cores, full read)")
    print("=" * 74)
    sub = [paired[i] for i in np.linspace(0, len(paired) - 1, 6).astype(int)]
    chan_stats = {c: [] for c in config.SRS_CHANNELS}
    for core in sub:
        arr = tifffile.imread(srs_files[core])
        if arr.ndim != 3:
            print(f"  {core}: UNEXPECTED ndim {arr.ndim}, shape {arr.shape}")
            continue
        # normalise to (channel, y, x)
        if arr.shape[0] != 4 and arr.shape[-1] == 4:
            arr = np.moveaxis(arr, -1, 0)
        line = f"  {core:6s} {str(arr.shape):20s} {arr.dtype} "
        for ch in range(arr.shape[0]):
            a = arr[ch]
            s = dict(ch=ch, mn=int(a.min()), p50=float(np.percentile(a, 50)),
                     p999=float(np.percentile(a, 99.9)), mx=int(a.max()),
                     frac_zero=float((a == 0).mean()))
            chan_stats[ch].append(s)
            line += f"| ch{ch} {s['mn']}-{s['mx']} p50={s['p50']:.0f} "
        print(line)

    print("\n  per-channel medians across the 6 cores:")
    for ch, name in config.SRS_CHANNELS.items():
        st = chan_stats.get(ch, [])
        if not st:
            continue
        print(f"    ch{ch} {name:14s} max={np.median([s['mx'] for s in st]):8.0f} "
              f"p50={np.median([s['p50'] for s in st]):8.0f} "
              f"p99.9={np.median([s['p999'] for s in st]):8.0f} "
              f"zero-frac={np.median([s['frac_zero'] for s in st]):.3f}")

    # --- persist ------------------------------------------------------------
    out_csv = config.RESULTS_DIR / "00_inventory.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_srs": len(srs_cores), "n_he": len(he_cores), "n_paired": len(paired),
        "paired": paired, "srs_only": srs_only, "he_only": he_only,
        "srs_placeholders": srs_cloud, "he_placeholders": he_cloud,
        "he_naming": {k: sorted(v, key=core_sort_key) for k, v in he_naming.items()},
        "srs_shapes": srs_shapes,
        "he_shapes": {f"{k[0]} {k[1]}": v for k, v in he_shapes.items()},
        "disagreements_with_brief": disagreements,
    }
    (config.RESULTS_DIR / "00_inventory.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("DISAGREEMENTS WITH HANDOVER.md")
    print("=" * 74)
    if disagreements:
        for d in disagreements:
            print(f"  ! {d}")
    else:
        print("  none on cohort composition")
    print(f"\n  wrote {out_csv}")
    print(f"  wrote {config.RESULTS_DIR / '00_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
