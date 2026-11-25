# Pre-specified analyses (verbatim excerpts from the dated analysis log)

The project kept an append-only analysis log (`DECISION_LOG.md`, not released in full because it
also records exploratory work not reported in the paper). The entries below were written BEFORE the
results they govern were computed and are reproduced verbatim. Internal arm names: A1 = CG+M+A,
A5 = CG+M+C. "Pre-registered" in the log means pre-specified in this dated log, not a public
registry.

Timing of the confirmatory test (D-19): the entry carries no date of its own; in the log it sits
after the 2026-09-26 seed-replication verdict and before the 2026-09-28 verdict that applies it.
The criterion was written after two seeds (42, 7) of both models had been evaluated and before the
third seed (123) was trained; the test pools all three seeds. Primary criterion: disc AND log-log slope, three-seed mean difference with a core-cluster
bootstrap 95% CI excluding 0; secondary: per-patch r and dense-class ratio.

---

### D-19 Third seed of A5 and A1 (running)

The main positive claim (A5 > A1 on conditioning) rests on a post hoc
pooled two-seed analysis. A third seed (123) of both arms, identical
otherwise. **Pre-registered, confirmatory:** on the pooled held-out set
(17 cores), the three-seed mean A5 - A1 difference in disc AND in log-log
slope has a core-cluster bootstrap 95% CI excluding 0. Secondary: the same
for per-patch r and high-density ratio.

---


---

### D-29 Stage A of the s18 screen: pre-specification (written before any Stage A result, 2026-10-01)
Owner decision: option D, Stage A only (epoch 30, 12 runs, pooled held-out 1038 patches, five detectors). No encoders, no paper figures; stop after Stage A.

Pipeline (s18lib.py, s18f_stageA_images.py, s18g_stageA_detect.py, s18i_stageA_report.py). Every image (real original, real round trip "rt", fake) is measured from its RGB: stain OD = skimage separate_stains with v1's calibrated hed_from_rgb (the call inside rgb_to_he). Check: on fakes this reproduces the generator's H to 0.6% (mean |dH| 0.0004 OD), so no H/E arrays are saved. rgb_to_he clips H at 0.125 OD, which binds fakes and rt but not original reals; hence both real baselines for every metric. Smoke of the table pipeline (12 patches x 14 image sets x 5 detectors): StarDist fake counts equal the existing epoch-30 eval CSV n_fake on 48/48 patches and real counts equal the cache on 12/12.
Tissue (M5, territories): smoothed summed RGB OD > 0.20, components and holes < 200 px removed; identical rule for all images. Visually correct; very pale fake cytoplasm can fall below it (limitation, not tuned).
Stromal patch: real StarDist count <= 10th percentile of the pooled set (23) and real tissue fraction >= 0.5. Density class: v1 buckets on the real StarDist count (original RGB), the same classes for every detector.
Saved: per-nucleus tables (patch_uid, centroid, area um2, solidity, eccentricity, mean H, mean E) and per-patch tables for every detector x image set; RGB PNGs kept only for a 40-patch stratified gallery (14/13/13), the rest deleted after Stage A.

Metrics (patch level; model-level value = median over patches of fake/real per patch, mean over the 3 seeds; CIs = core-cluster bootstrap n=1000, cores resampled jointly across seeds):
M1 median nuclear area; also by density class. M2 median solidity, median eccentricity (StarDist, CellViT only). M3 mean nearest-neighbour distance; median cell territory (tissue-restricted discrete Voronoi, nuclei >= 20 px from the border). M4 median over nuclei of mean nuclear H OD; also by density class. M5 mean H, mean E in tissue, H fraction H/(H+E); by density class and stromal patches (detector-free). M6 per detector: Pearson r, log-log slope, disc, as in s05 (fake count vs that detector's real count). R1 per detector: rt/original count (median ratio), plus M1/M4/M5 rt/original.
Baseline: every fake/real ratio is reported against BOTH real original and real rt. Primary for criterion (b) = rt (identical processing).
Eligibility from the smoke test: Cellpose-SAM counts only (M6, R1); InstanSeg counts and spacing (M3, M6, R1); HoVer-Net all except M2, flagged for R1 and fragmentation; StarDist and CellViT-SAM-H all.

Pass rules (fixed now):
(a) Agreement with StarDist on the 1038 real original patches: counts r >= 0.90 and log-log slope in [0.85, 1.15]; per-patch metric values (M1-M4) r >= 0.70. StarDist = reference (pass by definition). M5 detector-free (n/a).
R1 gate (detector): median rt/original count in [0.95, 1.05].
(b) Seed variance small relative to between-model spread: pooled within-model seed SD / SD of the four model means <= 0.5 (12 run values, primary rt baseline, overall summary of each metric).
(c) Not dominated by detection failure on pale fakes (all 12,456 fakes): empty fraction <= 1%, fraction with < 5 nuclei <= 5%; for second detectors also (detector/StarDist count) in the palest tercile of fake patches (by tissue H p99) / darkest tercile >= 0.85. Metrics needing >= 2 nuclei: report the undefined fraction.
(d) Cost: median detector time <= 0.5 s/image (Stage B 49,824 images in <= ~7 h per detector).
A metric x detector row passes only if all applicable criteria pass. Which model a metric favours is not a criterion. All rows reported.


---

### D-31 Stage B of the s18 screen: pre-specification (written before any Stage B result, 2026-10-02)
Owner approved the D-30 recommendation ("Ok - let's do it"; confidence 0.75 that this means the recommended scope: CellViT-SAM-H and InstanSeg, M6 only, epochs 27-30). Stage B = epochs 27, 28, 29 newly generated and detected; epoch 30 reused from Stage A. No encoders, no paper figures, no M1-M5 (kept descriptive from Stage A at epoch 30).
Pipeline: s18f_stageA_images.py --epoch N --out stageB_eN (same code, same 12 runs, same pooled held-out 1038 patches; real and rt images regenerated but not re-detected, Stage A real/rt tables are used); s18g_stageA_detect.py --det {cellvit_sam,instanseg} --root ...\stageB_eN --sets <12 run tags>; launcher run_s18_stageB.ps1. Stage B images are all temporary and deleted after the tables are checked (no gallery).
Metric: per run, the per-patch fake count is averaged over epochs 27-30 (as s05 mean_over_ckpts / s10b pooled), then M6 r, log-log slope, disc against that detector's real (rt) count, exactly as Stage A. Run values -> pass rule (b): pooled within-model seed SD / SD of the four model means <= 0.5. Criteria (a), R1 and (d) carry over from Stage A (detector-level, not recomputed; (d) re-checked on the new timings). (c) re-checked on the new fakes (empty <= 1%, <5 nuclei <= 5%).
Also reported: the epoch 27-30 StarDist M6 rows from the existing eval CSVs (reference, not re-run), and each detector's agreement with the model ordering. Which model a metric favours remains not a criterion.
Expected cost: generation about 11 min per epoch; CellViT about 0.15-0.25 s/img, InstanSeg 0.03 s/img plus CPU measurement; 37,368 images per detector; 3.5-4 h in total.
