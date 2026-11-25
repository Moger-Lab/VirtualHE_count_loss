"""
s18i -- Stage A report for the s18 metric/detector screen (vhe-detectors env).
Applies the rules pre-specified in DECISION_LOG D-29 to the tables written by
s18g. Screening output only: no paper figures. Never overwrites.

Outputs (outputs\\s18_metric_screen\\stageA\\):
  screen_table.csv          ONE table: every metric x detector x summary row,
                            model values (rt and original baselines, CIs),
                            criteria a-d, R1 gate, PASS/FAIL
  per_run_values.csv        12 run values for every row (both baselines)
  by_density_class.csv      M1, M4, M5 by density class (+ M5 stromal)
  detector_checks.csv       per detector: agreement on real, R1, empty /
                            near-empty fakes, paleness ratio, cost
  sanity_stardist_vs_eval.csv  StarDist epoch-30 counts vs the existing eval CSVs
  fig_*.png                 diagnostic plots
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd

ROOT = os.environ.get("S18_ROOT", __import__("vhe_paths").work("s18_metric_screen", "stageA"))
WORK_RUNS = __import__("vhe_paths").work("runs")
OUT = os.environ.get("S18_OUT", os.path.join(_script_dir, "outputs", "s18_metric_screen", "stageA"))
MODELS = {"CG": ["CG_full", "CG_full_s7", "CG_full_s123"],
          "CG+M": ["CGM_full", "CGM_full_s7", "CGM_full_s123"],
          "CG+M+A": ["A1_full", "A1_full_s7", "A1_full_s123"],
          "CG+M+C": ["A5_full", "A5_full_s7", "A5_full_s123"]}
COLORS = {"CG": "#7a5bb5", "CG+M": "#b5791c", "CG+M+A": "#a09e97", "CG+M+C": "#2a78d6"}
MARKERS = {"CG": "o", "CG+M": "s", "CG+M+A": "D", "CG+M+C": "^"}
RUNS = [t for v in MODELS.values() for t in v]
DETS = ["stardist", "cellvit_sam", "hovernet", "instanseg", "cellpose"]
DNAME = {"stardist": "StarDist", "cellvit_sam": "CellViT-SAM-H", "hovernet": "HoVer-Net",
         "instanseg": "InstanSeg", "cellpose": "Cellpose-SAM", "none": "(detector-free)"}
BUCKETS = ("low", "med", "high")
METRIC_COLS = {  # family, column, needs >= 2 nuclei
    "M1 nuclear area": ("M1", "med_area_um2"),
    "M2 solidity": ("M2", "med_solidity"),
    "M2 eccentricity": ("M2", "med_eccentricity"),
    "M3 nearest-neighbour distance": ("M3", "mean_nnd_um"),
    "M3 cell territory": ("M3", "med_territory_um2"),
    "M4 nuclear H OD": ("M4", "med_nuc_H"),
}
M5_COLS = {"M5 mean H (tissue)": "mean_H_tissue", "M5 mean E (tissue)": "mean_E_tissue", "M5 H fraction": "H_frac"}
ELIG = {"stardist": {"M1", "M2", "M3", "M4", "M6"}, "cellvit_sam": {"M1", "M2", "M3", "M4", "M6"},
        "hovernet": {"M1", "M3", "M4", "M6"}, "instanseg": {"M3", "M6"}, "cellpose": {"M6"}}
NOTES = {"hovernet": "smoke: fragments dense nuclei", "cellpose": "counts only (outlines whole cells)",
         "instanseg": "counts and spacing only (blocky outlines)"}
# D-29 thresholds
A_COUNT_R, A_SLOPE, A_METRIC_R = 0.90, (0.85, 1.15), 0.70
R1_GATE = (0.95, 1.05)
B_RATIO = 0.5
C_EMPTY, C_LT5, C_PALE = 0.01, 0.05, 0.85
D_SEC = 0.5
N_BOOT = 1000
STAGE_B_IMAGES = 12 * 4 * 1038 + 2 * 1038


def load():
    pt = pd.read_csv(os.path.join(ROOT, "patches.csv"))
    T = {}
    for d in DETS:
        for s in ["real", "rt"] + RUNS:
            p = pd.read_csv(os.path.join(ROOT, "tables", d, f"{s}_patches.csv"))
            assert list(p.patch_uid) == list(pt.patch_uid), (d, s)
            T[(d, s)] = p
    return pt, T


def slope(x, y):
    m = (x > 0) & (y > 0)
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def m6(nf, nr, bucket):
    """r, slope, disc as in s05 (ratio = fake / max(real, 1)); buckets fixed
    by the real StarDist count."""
    ratio = nf / np.maximum(nr, 1)
    lo, hi = np.median(ratio[bucket == "low"]), np.median(ratio[bucket == "high"])
    return dict(r=float(np.corrcoef(nr, nf)[0, 1]), slope=slope(nr, nf), disc=float(hi / max(lo, 1e-9)))


def within_between(vals):
    """vals: dict model -> 3 run values. (within SD, between SD, ratio)."""
    w = np.sqrt(np.mean([np.var(v, ddof=1) for v in vals.values()]))
    b = np.std([np.mean(v) for v in vals.values()], ddof=1)
    return w, b, (w / b if b > 0 else np.inf)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "screen_table.csv")):
        sys.exit(f"{OUT} already has results; refusing to overwrite")
    pt, T = load()
    bucket = pt.bucket.values
    stromal = pt.stromal.values.astype(bool)
    cores = pt.core.values
    core_idx = {c: np.nonzero(cores == c)[0] for c in np.unique(cores)}
    ucores = np.array(sorted(core_idx))
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([core_idx[c] for c in rng.choice(ucores, len(ucores))]) for _ in range(N_BOOT)]
    print(f"{len(pt)} patches, {len(ucores)} cores, {stromal.sum()} stromal")

    # ---------------- sanity: StarDist vs existing eval CSVs (epoch 30) ----------------
    san = []
    for t in RUNS:
        e = pd.concat([pd.read_csv(os.path.join(WORK_RUNS, t, f"G_S2H_epoch030_{s}_eval.csv")) for s in ("val", "legacy_test")])
        m = T[("stardist", t)].merge(e[["patch_uid", "n_fake", "n_real"]], on="patch_uid")
        san.append(dict(run=t, patches=len(m), count_mismatch=int((m.n != m.n_fake).sum()),
                        real_mismatch=int((T[("stardist", "real")].n.values != pt.n_real.values).sum())))
    san = pd.DataFrame(san); san.to_csv(os.path.join(OUT, "sanity_stardist_vs_eval.csv"), index=False)
    print("sanity StarDist vs eval CSV ep30: total count mismatches", int(san.count_mismatch.sum()))

    # ---------------- detector-level checks ----------------
    sd_real = T[("stardist", "real")].n.values.astype(float)
    pale = np.concatenate([T[("stardist", t)].H_p99_tissue.values for t in RUNS])
    q1, q2 = np.nanpercentile(pale, [100 / 3, 200 / 3])
    det_rows = []
    for d in DETS:
        real = T[(d, "real")]; rt = T[(d, "rt")]
        nr = real.n.values.astype(float)
        rec = dict(detector=d, name=DNAME[d], eligible="+".join(sorted(ELIG[d])), note=NOTES.get(d, ""))
        rec["a_count_r"] = float(np.corrcoef(sd_real, nr)[0, 1])
        rec["a_count_slope"] = slope(sd_real, nr)
        rec["real_level_vs_sd"] = float(np.median(nr / np.maximum(sd_real, 1)))
        ok = nr > 0
        rec["R1_rt_over_orig_count"] = float(np.median(rt.n.values[ok] / nr[ok]))
        rec["R1_rt_minus_orig_mean"] = float((rt.n.values - nr).mean())
        nf = np.concatenate([T[(d, t)].n.values for t in RUNS]).astype(float)
        nsd = np.concatenate([T[("stardist", t)].n.values for t in RUNS]).astype(float)
        rec["c_fake_min_n"] = int(nf.min()); rec["c_frac_empty"] = float((nf == 0).mean())
        rec["c_frac_lt5"] = float((nf < 5).mean()); rec["c_frac_lt2"] = float((nf < 2).mean())
        rr = nf / np.maximum(nsd, 1)
        rec["c_pale_ratio"] = (float(np.median(rr[pale <= q1]) / np.median(rr[pale > q2])) if d != "stardist" else np.nan)
        secs = np.concatenate([T[(d, s)].sec.values for s in ["real", "rt"] + RUNS])
        rec["d_sec_median"] = float(np.median(secs))
        rec["d_stageB_hours"] = float(np.median(secs) * STAGE_B_IMAGES / 3600)
        rec["pass_a_count"] = d == "stardist" or (rec["a_count_r"] >= A_COUNT_R and A_SLOPE[0] <= rec["a_count_slope"] <= A_SLOPE[1])
        rec["pass_R1"] = R1_GATE[0] <= rec["R1_rt_over_orig_count"] <= R1_GATE[1]
        rec["pass_c"] = (rec["c_frac_empty"] <= C_EMPTY and rec["c_frac_lt5"] <= C_LT5
                         and (d == "stardist" or rec["c_pale_ratio"] >= C_PALE))
        rec["pass_d"] = rec["d_sec_median"] <= D_SEC
        rec["detector_pass"] = rec["pass_a_count"] and rec["pass_R1"] and rec["pass_c"] and rec["pass_d"]
        det_rows.append(rec)
    DC = pd.DataFrame(det_rows).set_index("detector")
    DC.to_csv(os.path.join(OUT, "detector_checks.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print(DC.drop(columns=["name", "note"]).T.to_string())

    # ---------------- rows: metric x detector x summary ----------------
    rows, perrun, dens = [], [], []

    def summarise(label, family, det, getter, kind):
        """getter(run, base) -> per-patch array (ratio) or tuple for M6;
        kind 'ratio' (median of per-patch ratios) or an M6 key."""
        out = dict(metric=label, family=family, detector=det, name=DNAME[det])
        for base in ("rt", "real"):
            vals = {}
            arrs = {t: getter(t, base) for t in RUNS}
            for m, tags in MODELS.items():
                if kind == "ratio":
                    vals[m] = [float(np.nanmedian(arrs[t])) for t in tags]
                else:
                    vals[m] = [m6(*arrs[t], bucket)[kind] for t in tags]
                for t, v in zip(tags, vals[m]):
                    perrun.append(dict(metric=label, detector=det, baseline=base, model=m, run=t, value=v))
            # core-cluster bootstrap, cores resampled jointly across seeds
            bs = {m: [] for m in MODELS}
            for idx in boot_idx:
                for m, tags in MODELS.items():
                    if kind == "ratio":
                        bs[m].append(np.mean([np.nanmedian(arrs[t][idx]) for t in tags]))
                    else:
                        bs[m].append(np.mean([m6(arrs[t][0][idx], arrs[t][1][idx], bucket[idx])[kind] for t in tags]))
            sfx = "" if base == "rt" else "_orig"
            for m in MODELS:
                out[f"{m}{sfx}"] = float(np.mean(vals[m]))
                out[f"{m}{sfx}_lo"], out[f"{m}{sfx}_hi"] = np.percentile(bs[m], [2.5, 97.5])
            w, b, r = within_between(vals)
            out[f"seed_sd{sfx}"], out[f"between_sd{sfx}"], out[f"b_ratio{sfx}"] = w, b, r
        rows.append(out)
        return out

    def ratio_getter(det, col, subset=None):
        def g(t, base):
            f = T[(det, t)][col].values.astype(float); r = T[(det, base)][col].values.astype(float)
            x = np.where((r > 0) & np.isfinite(r) & np.isfinite(f), f / np.where(r > 0, r, 1), np.nan)
            if subset is not None:
                x = np.where(subset, x, np.nan)
            return x
        return g

    for d in DETS:
        for label, (fam, col) in METRIC_COLS.items():
            if fam in ELIG[d]:
                print(f"  {DNAME[d]:14s} {label}", flush=True)
                summarise(label, fam, d, ratio_getter(d, col), "ratio")
        for k in ("r", "slope", "disc"):
            print(f"  {DNAME[d]:14s} M6 {k}", flush=True)
            summarise(f"M6 density {k}", "M6", d,
                      lambda t, base, d=d: (T[(d, t)].n.values.astype(float), T[(d, base)].n.values.astype(float)), k)
    for label, col in M5_COLS.items():
        print(f"  M5 {label}", flush=True)
        o = summarise(label, "M5", "stardist", ratio_getter("stardist", col), "ratio")
        o["detector"] = "none"; o["name"] = DNAME["none"]

    # density classes (point estimates, mean over seeds), both baselines
    for d in DETS:
        for label, (fam, col) in METRIC_COLS.items():
            if fam in ("M1", "M4") and fam in ELIG[d]:
                for base in ("rt", "real"):
                    g = ratio_getter(d, col)
                    for m, tags in MODELS.items():
                        for bk in BUCKETS:
                            dens.append(dict(metric=label, detector=d, baseline=base, model=m, subset=bk,
                                             value=float(np.mean([np.nanmedian(g(t, base)[bucket == bk]) for t in tags]))))
    for label, col in M5_COLS.items():
        g = ratio_getter("stardist", col)
        for base in ("rt", "real"):
            for m, tags in MODELS.items():
                for name, msk in [(b, bucket == b) for b in BUCKETS] + [("stromal", stromal)]:
                    dens.append(dict(metric=label, detector="none", baseline=base, model=m, subset=name,
                                     value=float(np.mean([np.nanmedian(g(t, base)[msk]) for t in tags]))))
    # absolute real and fake M5 values too (H fraction etc. are interpretable on their own)
    for label, col in M5_COLS.items():
        for s, msk_name in [("real", "real orig"), ("rt", "real rt")] + [(t, t) for t in RUNS]:
            v = T[("stardist", s)][col].values
            for name, msk in [(b, bucket == b) for b in BUCKETS] + [("stromal", stromal), ("all", np.ones(len(pt), bool))]:
                dens.append(dict(metric=label + " (absolute)", detector="none", baseline="-", model=msk_name,
                                 subset=name, value=float(np.nanmedian(v[msk]))))

    S = pd.DataFrame(rows)
    # metric agreement with StarDist on real (criterion a for M1-M4)
    def a_metric(row):
        if row.family in ("M5",):
            return np.nan
        if row.family == "M6":
            return np.nan
        col = METRIC_COLS[row.metric][1]
        x = T[("stardist", "real")][col].values; y = T[(row.detector, "real")][col].values
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1])
    S["a_metric_r_vs_sd"] = S.apply(a_metric, axis=1)

    def verdict(row):
        why = []
        if row.detector != "none":
            dc = DC.loc[row.detector]
            if not dc.pass_a_count: why.append("a:count")
            if not dc.pass_R1: why.append("R1")
            if not dc.pass_c: why.append("c")
            if not dc.pass_d: why.append("d")
            if row.family in ("M1", "M2", "M3", "M4") and row.detector != "stardist" and row.a_metric_r_vs_sd < A_METRIC_R:
                why.append("a:metric")
        if not (row.b_ratio <= B_RATIO):
            why.append("b")
        return pd.Series(dict(PASS=len(why) == 0, fails=";".join(why)))
    S = pd.concat([S, S.apply(verdict, axis=1)], axis=1)
    S.to_csv(os.path.join(OUT, "screen_table.csv"), index=False)
    pd.DataFrame(perrun).to_csv(os.path.join(OUT, "per_run_values.csv"), index=False)
    pd.DataFrame(dens).to_csv(os.path.join(OUT, "by_density_class.csv"), index=False)
    cols = ["metric", "name"] + list(MODELS) + ["b_ratio", "a_metric_r_vs_sd", "PASS", "fails"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3, "display.max_rows", 200):
        print(S[cols].to_string(index=False))

    # ---------------- plots ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # agreement on real + R1
    fig, ax = plt.subplots(1, 5, figsize=(19, 4))
    for i, d in enumerate(DETS):
        nr = T[(d, "real")].n.values; nrt = T[(d, "rt")].n.values
        ax[i].scatter(sd_real, nr, s=5, alpha=0.4, color="#2a78d6", label="real (original)")
        ax[i].scatter(sd_real, nrt, s=5, alpha=0.4, color="#b5791c", label="real (round trip)")
        mx = max(sd_real.max(), nr.max()); ax[i].plot([0, mx], [0, mx], color="#555", lw=0.8, ls="--")
        dc = DC.loc[d]
        ax[i].set_title(f"{DNAME[d]}\nr {dc.a_count_r:.3f}  slope {dc.a_count_slope:.2f}  R1 {dc.R1_rt_over_orig_count:.3f}", fontsize=9)
        ax[i].set_xlabel("StarDist count, real original"); ax[i].set_ylabel("detector count")
    ax[0].legend(fontsize=7, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_agreement_real.png"), dpi=110); plt.close(fig)

    # run values per row (rt baseline)
    PR = pd.DataFrame(perrun); PR = PR[PR.baseline == "rt"]
    keys = list(S[["metric", "detector"]].itertuples(index=False, name=None))
    nc = 6; nrw = int(np.ceil(len(keys) / nc))
    fig, ax = plt.subplots(nrw, nc, figsize=(3.1 * nc, 2.5 * nrw), squeeze=False)
    for k, (mt, d) in enumerate(keys):
        a = ax[k // nc, k % nc]
        sub = PR[(PR.metric == mt) & (PR.detector == ("stardist" if d == "none" else d))]
        for j, m in enumerate(MODELS):
            v = sub[sub.model == m].value.values
            a.scatter(np.full(len(v), j) + np.array([-0.12, 0, 0.12])[:len(v)], v, color=COLORS[m], marker=MARKERS[m], s=22)
            a.plot([j - 0.25, j + 0.25], [v.mean()] * 2, color="#333", lw=1.2)
        row = S[(S.metric == mt) & (S.detector == d)].iloc[0]
        a.set_title(f"{mt}\n{DNAME[d]}  b={row.b_ratio:.2f}  {'PASS' if row.PASS else 'fail: ' + row.fails}", fontsize=7.5)
        a.set_xticks(range(4), list(MODELS), fontsize=7); a.tick_params(axis="y", labelsize=7)
    for k in range(len(keys), nrw * nc):
        ax[k // nc, k % nc].axis("off")
    fig.suptitle("Run values (3 seeds per model, epoch 30), fake vs real round trip; b = seed SD / between-model SD", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.98)); plt.savefig(os.path.join(OUT, "fig_run_values.png"), dpi=100); plt.close(fig)

    # paleness robustness
    fig, ax = plt.subplots(figsize=(6, 4))
    tert = np.where(pale <= q1, 0, np.where(pale <= q2, 1, 2))
    nsd = np.concatenate([T[("stardist", t)].n.values for t in RUNS]).astype(float)
    for d in DETS[1:]:
        nf = np.concatenate([T[(d, t)].n.values for t in RUNS]).astype(float)
        rr = nf / np.maximum(nsd, 1)
        ax.plot(range(3), [np.median(rr[tert == i]) for i in range(3)], "-o", label=DNAME[d])
    ax.set_xticks(range(3), ["palest third", "middle", "darkest third"]); ax.set_ylabel("detector / StarDist count (fakes)")
    ax.set_title("Fake patches by paleness (tissue H p99), all 12 runs", fontsize=9); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_paleness.png"), dpi=110); plt.close(fig)

    # per-nucleus distributions (StarDist, CellViT): area, solidity, eccentricity, mean H
    feats = [("area_um2", "nuclear area (um^2)", (0, 80)), ("solidity", "solidity", (0.8, 1.0)),
             ("eccentricity", "eccentricity", (0, 1)), ("mean_H", "mean nuclear H OD", (0, 0.2))]
    fig, ax = plt.subplots(2, 4, figsize=(17, 7))
    for i, d in enumerate(["stardist", "cellvit_sam"]):
        nt = {s: pd.read_csv(os.path.join(ROOT, "tables", d, f"{s}_nuclei.csv.gz")) for s in ["real", "rt"] + RUNS}
        for j, (c, lab, rg) in enumerate(feats):
            a = ax[i, j]; bins = np.linspace(*rg, 60)
            a.hist(nt["real"][c], bins, density=True, histtype="step", color="#222", lw=1.8, label="real (orig)")
            a.hist(nt["rt"][c], bins, density=True, histtype="step", color="#222", lw=1.0, ls="--", label="real (rt)")
            for m, tags in MODELS.items():
                a.hist(pd.concat([nt[t][c] for t in tags]), bins, density=True, histtype="step", color=COLORS[m], lw=1.3, label=m)
            a.set_xlabel(lab, fontsize=8); a.set_title(DNAME[d], fontsize=9)
    ax[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_nucleus_distributions.png"), dpi=100); plt.close(fig)
    print("wrote", OUT)
