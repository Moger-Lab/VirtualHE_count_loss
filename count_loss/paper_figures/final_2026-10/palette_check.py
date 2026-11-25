"""
palette_check -- Python port of the dataviz skill's validate_palette.js
(categorical checks only), because node is not installed on HILBERT.
Same thresholds, same Machado-Oliveira-Fernandes (2009) severity-1.0 CVD
simulation, OKLab Delta E x 100. Handoff 2026-10-02, analysis 6.

Usage: python palette_check.py "#7a5bb5,#b5791c,#a09e97,#2a78d6" [--pairs all] [--surface #ffffff]
"""

import argparse
import itertools

import numpy as np

BAND = (0.43, 0.77)
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def lin(h):
    h = h.strip().lstrip("#")
    s = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def oklab_lin(c):
    r, g, b = c
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def lch(h):
    L, a, b = oklab_lin(lin(h))
    return L, float(np.hypot(a, b))


def delta_e(h1, h2, kind=None):
    f = (lambda h: np.clip(np.array(MACHADO[kind]) @ lin(h), 0, 1)) if kind else lin
    return 100 * float(np.linalg.norm(oklab_lin(f(h1)) - oklab_lin(f(h2))))


def contrast(a, b):
    y = lambda h: float(np.array([0.2126, 0.7152, 0.0722]) @ lin(h))
    hi, lo = sorted([y(a), y(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def validate(pal, pairs="all", surface="#ffffff"):
    rows = []
    for c in pal:
        L, C = lch(c)
        rows.append(dict(color=c, L=round(L, 3), C=round(C, 3), band_ok=BAND[0] <= L <= BAND[1],
                         chroma_ok=C >= CHROMA_FLOOR, contrast=round(contrast(c, surface), 2)))
    pl = list(itertools.combinations(range(len(pal)), 2)) if pairs == "all" else [(i, i + 1) for i in range(len(pal) - 1)]
    prs = []
    for i, j in pl:
        prs.append(dict(a=pal[i], b=pal[j], normal=delta_e(pal[i], pal[j]), protan=delta_e(pal[i], pal[j], "protan"),
                        deutan=delta_e(pal[i], pal[j], "deutan"), tritan=delta_e(pal[i], pal[j], "tritan")))
    return rows, prs


def verdict(rows, prs):
    cvd = min(min(p["protan"], p["deutan"]) for p in prs)
    nor = min(p["normal"] for p in prs)
    return dict(band=all(r["band_ok"] for r in rows), chroma=all(r["chroma_ok"] for r in rows),
                cvd_worst=round(cvd, 1), cvd=("pass" if cvd >= CVD_TARGET else "floor" if cvd >= CVD_FLOOR else "fail"),
                normal_worst=round(nor, 1), normal="pass" if nor >= NORMAL_FLOOR else "fail",
                contrast_relief=[r["color"] for r in rows if r["contrast"] < CONTRAST_MIN])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--pairs", default="all", choices=["all", "adjacent"])
    ap.add_argument("--surface", default="#ffffff")
    a = ap.parse_args()
    pal = [c.strip() for c in a.palette.split(",") if c.strip()]
    rows, prs = validate(pal, a.pairs, a.surface)
    for r in rows:
        print(r)
    for p in prs:
        print(f"{p['a']} vs {p['b']}: normal {p['normal']:.1f}  protan {p['protan']:.1f}  deutan {p['deutan']:.1f}  tritan {p['tritan']:.1f}")
    print(verdict(rows, prs))
