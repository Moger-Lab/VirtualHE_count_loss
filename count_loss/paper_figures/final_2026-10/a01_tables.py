"""
a01 -- ONE four-model table on the real round-trip (rt) baseline, for every
figure and the text (handoff 2026-10-02, analysis 2; DECISION_LOG D-34).

Per run: per-patch StarDist fake count averaged over epochs 27-30 (eval CSVs),
real = StarDist on the rt image (s18 stage A table), density buckets fixed
by the original-RGB real count (as s18). Model value = mean over 3 seeds.
95% CI = core-cluster bootstrap (n = 1000, cores resampled jointly across
seeds, indices built exactly as s18k). Paired differences are paired by seed.
Stromal fabrication = fake nuclei per 12.5 um cell that is empty in the real
section (original-RGB StarDist cache; eval CSVs).

Outputs (data\\): four_model_rt_table.csv, four_model_rt_per_run.csv,
four_model_rt_paired.csv, four_model_rt_vs_orig.csv. Never overwrites.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcommon as F

import numpy as np
import pandas as pd

METRICS = ["r", "slope", "disc", "fake_real", "ratio_high", "stromal_fab"]
COMPS = [  # name, a, b, pre-registered
    ("CG+M minus CG", "CG+M", "CG", False),
    ("CG+M+A minus CG+M", "CG+M+A", "CG+M", False),
    ("CG+M+C minus CG+M", "CG+M+C", "CG+M", False),
    ("CG+M+C minus CG+M+A", "CG+M+C", "CG+M+A", True),
    ("CG+M+C minus CG", "CG+M+C", "CG", False),
]


def metrics_all(pt, R, idx=None):
    bucket = pt.bucket.values
    nr = pt.n_real_rt.values
    out = {}
    for k in F.KEYS:
        for t in F.MODEL[k]["tags"]:
            r = R[t]
            if idx is None:
                out[t] = F.run_metrics(r.n_fake.values, nr, bucket, r.fake_in_empty.values, r.n_empty_cells.values)
            else:
                out[t] = F.run_metrics(r.n_fake.values[idx], nr[idx], bucket[idx],
                                       r.fake_in_empty.values[idx], r.n_empty_cells.values[idx])
    return out


if __name__ == "__main__":
    paths = {n: os.path.join(F.DATA, f"four_model_rt_{n}.csv") for n in ("table", "per_run", "paired", "vs_orig")}
    if any(os.path.exists(p) for p in paths.values()):
        sys.exit("a01 outputs exist; refusing to overwrite")
    pt, R = F.load_all()
    print(f"{len(pt)} patches, {pt.core.nunique()} cores")
    point = metrics_all(pt, R)
    boots = [metrics_all(pt, R, idx) for idx in F.boot_indices(pt)]

    per_run = [dict(model=k, run=t, seed=s, **point[t]) for k in F.KEYS for t, s in zip(F.MODEL[k]["tags"], F.SEEDS)]
    pd.DataFrame(per_run).to_csv(paths["per_run"], index=False)

    def mmean(m, k, src):
        return np.mean([src[t][m] for t in F.MODEL[k]["tags"]])

    rows = []
    for k in F.KEYS:
        row = dict(model=k, seeds=3, patches=len(pt), cores=pt.core.nunique())
        for m in METRICS:
            b = [mmean(m, k, bs) for bs in boots]
            row[m] = mmean(m, k, point)
            row[m + "_lo"], row[m + "_hi"] = np.percentile(b, [2.5, 97.5])
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(paths["table"], index=False)

    crow = []
    for name, a, b, prereg in COMPS:
        for m in METRICS:
            def d(src):
                return np.mean([src[ta][m] - src[tb][m] for ta, tb in zip(F.MODEL[a]["tags"], F.MODEL[b]["tags"])])
            bs = [d(x) for x in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            crow.append(dict(comparison=name, a=a, b=b, pre_registered=prereg, metric=m, diff=d(point),
                             ci_lo=lo, ci_hi=hi, excl_0=bool(lo > 0 or hi < 0)))
    P = pd.DataFrame(crow)
    P.to_csv(paths["paired"], index=False)

    # reconciliation with the 2026-10-01 original-RGB table and the s18 stage B table
    old = pd.read_csv(os.path.join(F.PROJECT, "outputs", "s06_full", "four_model_table.csv"))
    old["model"] = old.model.str.replace(r" \(A[15]\)", "", regex=True)
    old = old.set_index("model")
    sb = pd.read_csv(os.path.join(F.PROJECT, "outputs", "s18_metric_screen", "stageB", "stageB_table.csv"))
    sb = sb[sb.detector == "stardist"].set_index("metric")
    rec = []
    for k in F.KEYS:
        t = T.set_index("model").loc[k]
        for m, om in (("r", "r"), ("slope", "slope"), ("disc", "disc"), ("fake_real", "fake/real"),
                      ("ratio_high", "high_ratio"), ("stromal_fab", "stromal_fab")):
            r = dict(model=k, metric=m, rt=t[m], orig_2026_10_01=old.loc[k, om], diff_vs_orig=t[m] - old.loc[k, om])
            if m in ("r", "slope", "disc"):
                s = sb.loc[f"M6 density {m}"]
                r.update(s18B=s[k], s18B_lo=s[k + "_lo"], s18B_hi=s[k + "_hi"],
                         diff_vs_s18B=t[m] - s[k], ci_diff_vs_s18B=max(abs(t[m + "_lo"] - s[k + "_lo"]), abs(t[m + "_hi"] - s[k + "_hi"])))
            rec.append(r)
    pd.DataFrame(rec).to_csv(paths["vs_orig"], index=False)

    with pd.option_context("display.width", 220, "display.precision", 3, "display.max_columns", 30):
        print(T[["model"] + [c for m in METRICS for c in (m, m + "_lo", m + "_hi")]].to_string(index=False))
        print()
        print(P[P.metric.isin(["r", "slope", "disc", "stromal_fab"])].to_string(index=False))
        print()
        print(pd.DataFrame(rec).to_string(index=False))
