"""
a05 -- share of a count loss carried by the patch-total term vs the per-cell
term for a uniform 40% undercount on training patches (StarDist targets), at
12.5 um and 50 um cells (Supplementary Fig. S8b). Extracted verbatim from the
project's earlier figure script (function loss_shares); needs the StarDist
cache (work/stardist_v2). Writes paper_figures/out/fig4_global_vs_local_d_loss_shares_source.csv,
the file figS8_design.py reads (the released copy is already there).
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, PROJECT)

import numpy as np
import pandas as pd
import torch

import countlib as cl

def loss_shares(n_patches=800, under=0.4, seed=0):
    sd = pd.read_csv(os.path.join(PROJECT, "outputs", "s02_counter", "stardist_per_patch.csv"))
    tr = sd[(sd.split == "train") & (sd.role == "train")].sample(n_patches, random_state=seed)
    sc = cl.stardist_cache()
    rows = []
    for grid in (8, 2):
        cells = np.stack([cl.cell_counts_from_centroids(sc[u][0], sc[u][1], grid) for u in tr.patch_uid])
        q95 = np.percentile(cells, 95)
        Mr = torch.from_numpy(cells).float()
        # build per-cell maps directly as counts (sum_pool of a map whose cells hold these totals)
        n_r = Mr; n_f = Mr * (1 - under)
        e = (n_f - n_r) / torch.sqrt(n_r + 1)
        rho = torch.clamp(n_r / q95, 0, 1)
        w = torch.where(e > 0, 3 * (1 - rho) + rho, (1 - rho) + 3 * rho)
        hub = torch.nn.functional.smooth_l1_loss(e, torch.zeros_like(e), beta=1.0, reduction="none")
        L_cell = (hub * w).mean(dim=(1, 2))
        N_r = n_r.sum(dim=(1, 2)); N_f = n_f.sum(dim=(1, 2))
        E = (N_f - N_r) / torch.sqrt(N_r + 1)
        L_tot = 0.5 * torch.nn.functional.smooth_l1_loss(E, torch.zeros_like(E), beta=1.0, reduction="none")
        rows.append(dict(cell_um=100 / grid, grid=grid, mean_real_per_cell=float(cells.mean()),
                         L_cell=float(L_cell.mean()), L_total_half=float(L_tot.mean()),
                         total_share=float((L_tot / (L_cell + L_tot)).mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = loss_shares()
    out = os.path.join(PROJECT, "paper_figures", "out", "fig4_global_vs_local_d_loss_shares_source.csv")
    if os.path.exists(out):
        sys.exit(f"{out} exists; refusing to overwrite")
    t.to_csv(out, index=False)
    print(t.to_string(index=False))
