"""Supplementary Fig S10 -- the typical held-out core (D-2) in the format of Fig 4d."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F
import core_panels as CP

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if __name__ == "__main__":
    if not os.path.isdir(CP.IMGC):
        sys.exit("img_cores not found (core mosaics are not distributed); S10 skipped")
    F.apply()
    pt, R = F.load_all()
    fig = plt.figure(figsize=(F.DOUBLE, 112 * F.MM))
    gs = GridSpec(1, 1, figure=fig, left=0.07, right=0.99, top=0.95, bottom=0.10)
    cor, pcn = CP.build(fig, gs[0, 0], ["D-2"], pt.groupby("core").n_real.median())
    F.save(fig, "figS10_typical_core", {"cores": cor, "patch_counts": pcn})
