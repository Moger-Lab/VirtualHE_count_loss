"""
g04 -- nucleus masks, stain maps and empty-cell data for the explanatory panels
of Figs 3 and 5 (srs-analysis env, GPU not needed).

Patches: the four Fig 2 patches (sparse, intermediate, dense, stromal; seed 42,
epoch 30), images from D:\\vhe_count_work\\paper_figs_2026-10\\img. For each
patch and each image set (real original RGB, real round trip "rt", CG, CG+M,
CG+M+A, CG+M+C) StarDist (v1 protocol, countlib.detect_nuclei) is run on the
512 px RGB; labels with area < MIN_NUCLEUS_AREA_PX are dropped (so the number
of labels is the count used everywhere).

Writes (never overwrites): data\\example_labels\\<uid>_<set>.npz (labels, and for
the stromal patch also the calibrated H and E optical density maps and tissue
mask), D:\\...\\img\\rt\\<uid>.png (round-trip real), data\\fig_examples_checks.csv.

Checks against the evaluation tables: fake counts = eval CSV n_fake (seed 42,
epoch 30); real count = StarDist cache; rt count and every patch-level M1 / M5
value = stageA tables; fake_in_empty = eval CSV.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import fcommon as F
sys.path.insert(0, F.PROJECT)

import numpy as np
import pandas as pd

import countlib as cl
import s18lib as sl

IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
LAB = os.path.join(F.DATA, "example_labels")
TAGS = {"CG": "CG_full", "CG+M": "CGM_full", "CG+M+A": "A1_full", "CG+M+C": "A5_full"}
SETS = ["real", "rt"] + list(TAGS)
TABLES = os.path.join(F.STAGEA, "tables", "stardist")


def clean_labels(lab):
    from skimage.measure import regionprops
    out = np.zeros(lab.shape, np.uint16)
    k = 0
    for p in regionprops(lab):
        if p.area >= cl.MIN_NUCLEUS_AREA_PX:
            k += 1
            out[lab == p.label] = k
    return out


if __name__ == "__main__":
    import countnet as cn
    sel = pd.read_csv(os.path.join(F.DATA, "fig2_final_selection.csv"))
    uids = list(sel.patch_uid)
    stromal = sel[sel.row == "stromal"].patch_uid.iloc[0]
    os.makedirs(LAB, exist_ok=True)
    os.makedirs(os.path.join(IMG, "rt"), exist_ok=True)
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    sc = cl.stardist_cache()
    ev = {t: pd.concat([pd.read_csv(os.path.join(F.RUNS_DIR, t, f"G_S2H_epoch030_{s}_eval.csv"))
                        for s in ("val", "legacy_test")]).set_index("patch_uid") for t in TAGS.values()}
    tab = {s: pd.read_csv(os.path.join(TABLES, f"{n}_patches.csv")).set_index("patch_uid")
           for s, n in [("rt", "rt"), ("real", "real")] + list(TAGS.items())}

    rec = []
    for u in uids:
        real = sl.read_png(os.path.join(IMG, "real", f"{u}.png"))
        p = os.path.join(IMG, "rt", f"{u}.png")
        if not os.path.exists(p):
            h = cn.v1_common.rgb_to_he(real, calib)
            sl.write_png(p, cn.v1_common.he_to_rgb(h[0], h[1], calib))
        ry, rx, ra = sc[u]
        for s in SETS:
            f = os.path.join(LAB, f"{u}_{s}.npz")
            if os.path.exists(f):
                sys.exit(f"{f} exists; refusing to overwrite")
            rgb = sl.read_png(os.path.join(IMG, {"real": "real", "rt": "rt"}.get(s, TAGS.get(s)), f"{u}.png"))
            (cy, cx, ar), lab = cl.detect_nuclei(rgb)
            lab = clean_labels(lab)
            n = int(lab.max())
            H, E = sl.stain_od(rgb)
            t = sl.tissue_mask(rgb)
            rec_s, _, _, _ = sl.stain_patch(rgb)
            area = float(np.median(np.bincount(lab.ravel())[1:])) * sl.UM2_PER_PX if n else np.nan
            ref_n = {"real": len(ra), "rt": tab["rt"].loc[u, "n"]}.get(s)
            if ref_n is None:
                ref_n = ev[TAGS[s]].loc[u, "n_fake"]
            tb = tab[s].loc[u]
            r = dict(patch_uid=u, set=s, n=n, n_expected=int(ref_n), med_area_um2=area, med_area_table=tb.med_area_um2,
                     meanH_tissue=rec_s["mean_H_tissue"], meanH_table=tb.mean_H_tissue,
                     meanE_tissue=rec_s["mean_E_tissue"], meanE_table=tb.mean_E_tissue)
            if s in TAGS:
                cf = cl.cell_counts_from_centroids(cy, cx, 8); cr = cl.cell_counts_from_centroids(ry, rx, 8)
                r["fake_in_empty"] = float(cf[cr == 0].sum()); r["fake_in_empty_eval"] = float(ev[TAGS[s]].loc[u, "fake_in_empty"])
                r["n_empty_cells"] = int((cr == 0).sum()); r["n_empty_eval"] = int(ev[TAGS[s]].loc[u, "n_empty_cells"])
            rec.append(r)
            extra = dict(H=H.astype(np.float16), E=E.astype(np.float16), tissue=t) if u == stromal else {}
            np.savez_compressed(f, labels=lab, **extra)
    C = pd.DataFrame(rec)
    C.to_csv(os.path.join(F.DATA, "fig_examples_checks.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(C.round(4).to_string(index=False))
    bad_n = int((C.n != C.n_expected).sum())
    bad_a = int((~np.isclose(C.med_area_um2, C.med_area_table, rtol=1e-3, equal_nan=True)).sum())
    bad_h = int((~np.isclose(C.meanH_tissue, C.meanH_table, rtol=1e-3, equal_nan=True)).sum())
    bad_f = int((~np.isclose(C.fake_in_empty, C.fake_in_empty_eval, equal_nan=True)).sum())
    print("count mismatches", bad_n, "| area mismatches", bad_a, "| meanH mismatches", bad_h, "| fake_in_empty mismatches", bad_f)
