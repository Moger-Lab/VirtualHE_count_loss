"""
s18j -- after Stage A: keep RGB images only for the 40-patch stratified
gallery (moved to stageA\\gallery\\<set>\\), delete the other temporary PNGs.
Refuses to run unless every detector x image-set table exists and has the
expected number of rows.

Usage: python s18j_stageA_cleanup.py [--dry_run]
"""

import argparse
import os
import shutil

import pandas as pd

ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageA")
DETS = ["stardist", "cellvit_sam", "hovernet", "instanseg", "cellpose"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    pt = pd.read_csv(os.path.join(ROOT, "patches.csv"))
    sets = sorted(os.listdir(os.path.join(ROOT, "img")))
    for d in DETS:
        for s in sets:
            p = os.path.join(ROOT, "tables", d, f"{s}_patches.csv")
            assert os.path.exists(p) and len(pd.read_csv(p)) == len(pt), f"incomplete: {p}"
            assert os.path.exists(os.path.join(ROOT, "tables", d, f"{s}_nuclei.csv.gz")), f"missing nuclei table {d} {s}"
    gal = set(pt.patch_uid[pt.gallery == 1])
    moved = deleted = 0
    for s in sets:
        src = os.path.join(ROOT, "img", s); dst = os.path.join(ROOT, "gallery", s)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            uid = f[:-4]
            if uid in gal:
                if not a.dry_run:
                    shutil.move(os.path.join(src, f), os.path.join(dst, f))
                moved += 1
            else:
                if not a.dry_run:
                    os.remove(os.path.join(src, f))
                deleted += 1
        if not a.dry_run and not os.listdir(src):
            os.rmdir(src)
    print(f"{'would move' if a.dry_run else 'moved'} {moved} gallery images, "
          f"{'would delete' if a.dry_run else 'deleted'} {deleted} temporary images")
