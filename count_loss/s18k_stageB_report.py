"""
s18k -- Stage B report for the s18 screen (vhe-detectors env). Rules: DECISION_LOG D-31.
M6 (density r, log-log slope, disc) for StarDist (existing eval CSVs, reference),
CellViT-SAM-H and InstanSeg, with the per-patch fake count averaged over
epochs 27-30. Criterion (b) = within-model seed SD / SD of the 4 model means <= 0.5.
Never overwrites. Outputs in outputs\\s18_metric_screen\\stageB\\.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd

from s18i_stageA_report import (MODELS, RUNS, COLORS, MARKERS, DNAME, m6, within_between, A_COUNT_R,
                                C_EMPTY, C_LT5, C_PALE, D_SEC, B_RATIO, N_BOOT)

A_ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageA")
B_ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageB_e{e}")
RUNS_DIR = __import__("vhe_paths").work("runs")
OUT = os.path.join(_script_dir, "outputs", "s18_metric_screen", "stageB")
EPOCHS = [27, 28, 29, 30]
DETS = ["stardist", "cellvit_sam", "instanseg"]
KEYS = ("r", "slope", "disc")


def counts(det, run, e, uids):
    if det == "stardist":
        d = pd.concat([pd.read_csv(os.path.join(RUNS_DIR, run, f"G_S2H_epoch{e:03d}_{s}_eval.csv"))
                       for s in ("val", "legacy_test")]).set_index("patch_uid")
        return d.loc[uids, "n_fake"].values.astype(float), None
    root = A_ROOT if e == 30 else B_ROOT.format(e=e)
    p = pd.read_csv(os.path.join(root, "tables", det, f"{run}_patches.csv"))
    assert list(p.patch_uid) == list(uids), (det, run, e)
    return p.n.values.astype(float), p


if __name__ == "__main__":
    if os.path.exists(os.path.join(OUT, "stageB_table.csv")):
        sys.exit(f"{OUT} already has results; refusing to overwrite")
    os.makedirs(OUT, exist_ok=True)
    pt = pd.read_csv(os.path.join(A_ROOT, "patches.csv"))
    uids = pt.patch_uid.values
    bucket = pt.bucket.values
    cores = pt.core.values
    core_idx = {c: np.nonzero(cores == c)[0] for c in np.unique(cores)}
    ucores = np.array(sorted(core_idx))
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([core_idx[c] for c in rng.choice(ucores, len(ucores))]) for _ in range(N_BOOT)]

    N, P = {}, {}
    for d in DETS:
        for t in RUNS:
            for e in EPOCHS:
                N[(d, t, e)], P[(d, t, e)] = counts(d, t, e, uids)
    base = {}
    for d in DETS:
        base[(d, "rt")] = pd.read_csv(os.path.join(A_ROOT, "tables", d, "rt_patches.csv")).n.values.astype(float)
        base[(d, "real")] = pd.read_csv(os.path.join(A_ROOT, "tables", d, "real_patches.csv")).n.values.astype(float)

    # sanity: StarDist from the eval CSVs equals Stage A StarDist at epoch 30
    mism = sum(int((N[("stardist", t, 30)] != pd.read_csv(os.path.join(A_ROOT, "tables", "stardist", f"{t}_patches.csv")).n.values).sum())
               for t in RUNS)
    print("StarDist eval CSV vs Stage A epoch-30 mismatches:", mism)

    # detector checks on the new epochs (c, d) for the second detectors
    dc = []
    sd_all = np.concatenate([N[("stardist", t, e)] for t in RUNS for e in EPOCHS])
    for d in DETS[1:]:
        ps = [P[(d, t, e)] for t in RUNS for e in EPOCHS]
        nf = np.concatenate([p.n.values for p in ps]).astype(float)
        pale = np.concatenate([p.H_p99_tissue.values for p in ps])
        q1, q2 = np.nanpercentile(pale, [100 / 3, 200 / 3])
        rr = nf / np.maximum(sd_all, 1)
        sec = np.concatenate([p.sec.values for p in ps])
        r = dict(detector=d, name=DNAME[d], images=len(nf), frac_empty=float((nf == 0).mean()),
                 frac_lt5=float((nf < 5).mean()), pale_ratio=float(np.median(rr[pale <= q1]) / np.median(rr[pale > q2])),
                 sec_median=float(np.median(sec)))
        r["pass_c"] = r["frac_empty"] <= C_EMPTY and r["frac_lt5"] <= C_LT5 and r["pale_ratio"] >= C_PALE
        r["pass_d"] = r["sec_median"] <= D_SEC
        dc.append(r)
    dc = pd.DataFrame(dc); dc.to_csv(os.path.join(OUT, "detector_checks_stageB.csv"), index=False)
    A_dc = pd.read_csv(os.path.join(_script_dir, "outputs", "s18_metric_screen", "stageA", "detector_checks.csv")).set_index("detector")
    print(dc.round(3).to_string(index=False))

    rows, perrun = [], []
    for d in DETS:
        for k in KEYS:
            o = dict(metric=f"M6 density {k}", detector=d, name=DNAME[d])
            for bs_name, sfx in (("rt", ""), ("real", "_orig")):
                nr = base[(d, bs_name)]
                for label, eps in (("ep30", [30]), ("ep27_30", EPOCHS)):
                    nf = {t: np.mean([N[(d, t, e)] for e in eps], axis=0) for t in RUNS}
                    vals = {m: [m6(nf[t], nr, bucket)[k] for t in tags] for m, tags in MODELS.items()}
                    for m, tags in MODELS.items():
                        for t, v in zip(tags, vals[m]):
                            perrun.append(dict(metric=o["metric"], detector=d, baseline=bs_name, checkpoints=label, model=m, run=t, value=v))
                    w, b, r = within_between(vals)
                    o[f"seed_sd_{label}{sfx}"], o[f"between_sd_{label}{sfx}"], o[f"b_ratio_{label}{sfx}"] = w, b, r
                    if label == "ep27_30":
                        bsamp = {m: [] for m in MODELS}
                        for idx in boot_idx:
                            for m, tags in MODELS.items():
                                bsamp[m].append(np.mean([m6(nf[t][idx], nr[idx], bucket[idx])[k] for t in tags]))
                        for m in MODELS:
                            o[f"{m}{sfx}"] = float(np.mean(vals[m]))
                            o[f"{m}{sfx}_lo"], o[f"{m}{sfx}_hi"] = np.percentile(bsamp[m], [2.5, 97.5])
            # checkpoint-to-checkpoint SD within a run (diagnostic)
            nr = base[(d, "rt")]
            esd = []
            for t in RUNS:
                esd.append(np.std([m6(N[(d, t, e)], nr, bucket)[k] for e in EPOCHS], ddof=1))
            o["epoch_sd_within_run"] = float(np.mean(esd))
            rows.append(o)
    S = pd.DataFrame(rows)

    def verdict(r):
        why = []
        if r.detector == "stardist":
            pass
        else:
            a = A_dc.loc[r.detector]
            if not a.pass_a_count: why.append("a:count")
            if not a.pass_R1: why.append("R1")
            c = dc.set_index("detector").loc[r.detector]
            if not c.pass_c: why.append("c")
            if not c.pass_d: why.append("d")
        if not (r.b_ratio_ep27_30 <= B_RATIO): why.append("b")
        return pd.Series(dict(PASS=len(why) == 0, fails=";".join(why)))
    S = pd.concat([S, S.apply(verdict, axis=1)], axis=1)
    S.to_csv(os.path.join(OUT, "stageB_table.csv"), index=False)
    pd.DataFrame(perrun).to_csv(os.path.join(OUT, "per_run_values.csv"), index=False)
    cols = ["metric", "name"] + list(MODELS) + ["b_ratio_ep30", "b_ratio_ep27_30", "epoch_sd_within_run", "PASS", "fails"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print(S[cols].to_string(index=False))

    # figure: run values (4-checkpoint mean) per detector x metric
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr = pd.DataFrame(perrun)
    fig, ax = plt.subplots(len(DETS), 3, figsize=(11, 8), sharex=True)
    for i, d in enumerate(DETS):
        for j, k in enumerate(KEYS):
            a = ax[i, j]
            for xi, (m, tags) in enumerate(MODELS.items()):
                for lab, dx, filled in (("ep30", -0.13, False), ("ep27_30", 0.13, True)):
                    v = pr[(pr.detector == d) & (pr.metric == f"M6 density {k}") & (pr.baseline == "rt")
                           & (pr.checkpoints == lab) & (pr.model == m)].value.values
                    a.scatter(np.full(len(v), xi + dx), v, marker=MARKERS[m], s=34, color=COLORS[m],
                              facecolors=COLORS[m] if filled else "none", linewidths=1.2)
            row = S[(S.detector == d) & (S.metric == f"M6 density {k}")].iloc[0]
            a.set_title(f"{DNAME[d]}  M6 {k}\nb ep30 {row.b_ratio_ep30:.2f}  b ep27-30 {row.b_ratio_ep27_30:.2f}  "
                        f"{'PASS' if row.PASS else 'fail: ' + row.fails}", fontsize=8)
            a.set_xticks(range(4)); a.set_xticklabels(list(MODELS), fontsize=8)
            a.grid(alpha=0.25)
    fig.suptitle("Stage B: run values vs real round trip (open = epoch 30 only, filled = mean of epochs 27-30; 3 seeds per model)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_stageB_run_values.png"), dpi=150)
    print("wrote", OUT)
