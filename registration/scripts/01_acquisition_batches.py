"""Stage 0b - is the cohort acquired under one configuration or several?

Two signals surfaced in the inventory:
  * the SRS TIFF ImageDescription takes 2 distinct values across 77 cores
  * channel 1 saturates at 4095 on A-1 but at ~2752 on every other sampled core

Both would indicate the cohort is not homogeneous. That matters because
HANDOVER.md section 7 records an unexplained density-independent per-core
effect of ~+/-17.5%, and because global (not per-patch) normalisation ranges
have to be fitted to whatever structure actually exists here.

Header reads only - fast, no bulk pixel I/O.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tifffile

import config
from importlib import import_module

core_sort_key = import_module("scripts.00_inventory").core_sort_key \
    if False else None  # noqa - avoid importing a numeric-prefixed module


def _sort_key(core: str):
    import re
    m = re.match(r"^([A-Za-z]+)-(\d+)$", core)
    return (m.group(1), int(m.group(2))) if m else (core, 0)


def parse_acquisition(desc: str) -> dict[str, str]:
    """Extract the [Acquisition Parameters] block.

    ImageDescription also embeds four 256-entry LUT tables. Those make every
    file's description textually unique and carry no acquisition information,
    so comparing whole descriptions is meaningless - only this block is.
    """
    out: dict[str, str] = {}
    inside = False
    for line in desc.split("\n"):
        line = line.strip()
        if line == "[Acquisition Parameters]":
            inside = True
            continue
        if line == "[Acquisition Parameters End]":
            break
        if inside and "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    srs = {p.name[: -len(config.SRS_SUFFIX)]: p
           for p in config.SRS_DIR.glob(f"*{config.SRS_SUFFIX}")}

    by_desc: dict[str, list[str]] = defaultdict(list)
    acq_rows = []
    tag_rows = []

    for core, path in sorted(srs.items(), key=lambda kv: _sort_key(kv[0])):
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = {t.name: t.value for t in page.tags}
            desc = (tags.get("ImageDescription") or "").strip()
            acq = parse_acquisition(desc)
            acq["core"] = core
            acq_rows.append(acq)
            # group on the acquisition block only, excluding the timestamp
            key = json.dumps({k: v for k, v in sorted(acq.items())
                              if k not in ("core", "Date", "Time")})
            by_desc[key].append(core)
            tag_rows.append({
                "core": core,
                "n_pages": len(tf.pages),
                "software": tags.get("Software"),
                "datetime": tags.get("DateTime"),
                "make": tags.get("Make"),
                "model": tags.get("Model"),
                "xres": str(tags.get("XResolution")),
                "resunit": str(tags.get("ResolutionUnit")),
                "bits": str(tags.get("BitsPerSample")),
            })

    print("=" * 74)
    print(f"ACQUISITION-PARAMETER VARIANTS (timestamp excluded): {len(by_desc)}")
    print("=" * 74)
    for i, (key, cores) in enumerate(
            sorted(by_desc.items(), key=lambda kv: -len(kv[1]))):
        cores_sorted = sorted(cores, key=_sort_key)
        print(f"\n--- variant {i}: {len(cores)} cores")
        print(f"    cores: {cores_sorted}")
        for k, v in json.loads(key).items():
            print(f"      {k:24s} = {v}")

    print("\n" + "=" * 74)
    print("ACQUISITION SESSIONS (by capture date)")
    print("=" * 74)
    by_date: dict[str, list[str]] = defaultdict(list)
    for r in acq_rows:
        by_date[r.get("Date", "?")].append(r["core"])
    for d, cores in sorted(by_date.items()):
        print(f"  {d}: {len(cores):3d} cores  {sorted(cores, key=_sort_key)}")

    print("\n" + "=" * 74)
    print("SCALE-RELEVANT ACQUISITION FIELDS")
    print("=" * 74)
    for field in ("Zoom Size", "Magnification", "Objective Lens",
                  "Scan Speed", "System Configuration", "ScanMode",
                  "Intensity Resolution"):
        vals = Counter(r.get(field, "<absent>") for r in acq_rows)
        if len(vals) == 1:
            print(f"  {field:22s} uniform: {next(iter(vals))}")
        else:
            print(f"  {field:22s} {len(vals)} distinct:")
            for v, n in vals.most_common():
                print(f"    {n:3d} x {v}")

    print("\n" + "=" * 74)
    print("OTHER TIFF TAGS (distinct value counts across cohort)")
    print("=" * 74)
    for field in ("n_pages", "software", "datetime", "make", "model",
                  "xres", "resunit", "bits"):
        vals = Counter(str(r[field]) for r in tag_rows)
        if len(vals) == 1:
            print(f"  {field:10s} uniform: {next(iter(vals))}")
        else:
            print(f"  {field:10s} {len(vals)} distinct:")
            for v, n in vals.most_common():
                print(f"               {n:3d} x {v[:90]}")

    out = config.RESULTS_DIR / "01_acquisition_batches.json"
    out.write_text(json.dumps(
        {"variants": {f"variant_{i}": {"cores": sorted(c, key=_sort_key),
                                       "params": json.loads(k)}
                      for i, (k, c) in enumerate(
                          sorted(by_desc.items(), key=lambda kv: -len(kv[1])))},
         "by_date": {d: sorted(c, key=_sort_key) for d, c in sorted(by_date.items())},
         "acquisition": acq_rows,
         "tags": tag_rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
