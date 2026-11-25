"""
final_selected -- the owner's choices from the candidate sets (2026-10-02,
DECISION_LOG D-37), written as the paper versions without candidate titles:

  fig2_examples              Fig 2 = candidate A (seed 42, epoch 30, all four
                             models, the fixed rule; CG seed 42 is CG's weakest
                             seed, stated in the legend)
  figS9_fig2_unselected      unselected counterpart of Fig 2 (candidate E,
                             random within each row class, seed 0)
  figS6b_cg_seeds            S6b = candidate A patch set (CG seeds 42 / 7 / 123)
  figS1b_registration        S1b = candidate A cores
Fig 1a core B-1 is set in fig1_schematic.py (D-36).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F
import fig2_candidates as G2
import figS6_cg_seeds as S6
import figS1_registration_examples as S1

import pandas as pd

SWAPS = {  # owner, 2026-10-02 (D-40): E-8 dense patch has SRS intensity banding -> next dense patch by the rule
    ("A", "dense"): "F-6_eval_2205_3087",
}


def apply_swaps(C):
    """Version rows with owner swaps applied; replacement rows rebuilt from the pool table."""
    P = pd.read_csv(os.path.join(F.DATA, "fig2_pool_quality.csv")).set_index("patch_uid")
    ren = {"n_real": "n_real_orig", "n_CG_full": "n_fake_CG", "n_CGM_full": "n_fake_CG+M", "n_A1_full": "n_fake_CG+M+A",
           "n_A5_full": "n_fake_CG+M+C", "n_CG_full_s7": "n_fake_CG_s7", "n_CG_full_s123": "n_fake_CG_s123"}
    C = C.copy()
    for (v, row), uid in SWAPS.items():
        i = C.index[(C.version == v) & (C.row == row)][0]
        p = P.loc[uid]
        C.loc[i, "patch_uid"] = uid
        C.loc[i, "rule"] = "owner swap (D-40): next dense patch in quality order"
        for k in ("core", "bucket", "stromal", "eligible", "focus", "tissue_iou", "dark_frac", "valid_frac"):
            C.loc[i, k] = p[k]
        for a, b in ren.items():
            C.loc[i, b] = p[a]
    return C


if __name__ == "__main__":
    F.apply()
    C = apply_swaps(pd.read_csv(os.path.join(F.DATA, "fig2_candidates.csv")))
    C[C.version == "A"].to_csv(os.path.join(F.DATA, "fig2_final_selection.csv"), index=False)
    cols_fig2 = list(G2.COLS)
    for v, name in (("A", "fig2_examples"), ("E (unselected)", "figS9_fig2_unselected")):
        G2.COLS = cols_fig2
        F.save(G2.build(C[C.version == v], None), name, {"patches": C[C.version == v]})
    G2.COLS = S6.COLS_S6
    F.save(G2.build(C[C.version == "A"], None), "figS6b_cg_seeds", {"patches": C[C.version == "A"]})
    S1C = pd.read_csv(os.path.join(F.DATA, "figS1_candidates.csv"))
    F.save(S1.build(S1C[S1C.version == "A"], None), "figS1b_registration", {"cores": S1C[S1C.version == "A"]})
