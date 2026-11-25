# s18 metric and detector screen: Stage A results (epoch 30, 2026-10-02)

Screening only; no existing result or figure changed; no paper figures; no encoders (M7 pending access).
Pre-specification: DECISION_LOG D-29 (written before any Stage A result). Outcome: D-30.

**Scope.** 12 runs (CG, CG+M, CG+M+A = A1, CG+M+C = A5; seeds 42/7/123), epoch 30 only, pooled held-out set
(17 cores, 1038 patches; 109 stromal). 14,532 images (1038 real, 1038 real round trip, 12,456 fakes) x 5 detectors.
Model value = mean over seeds of the median per-patch fake/real ratio; 95% CIs by core-cluster bootstrap
(n = 1000, cores resampled jointly across seeds). Primary real baseline = real after the rgb_to_he / he_to_rgb
round trip ("rt", identical processing to fakes); the original-RGB baseline is reported alongside and differs by
at most 0.04 in any model value (`screen_table.csv`, columns `*_orig`).

**Sanity.** The new StarDist pipeline reproduces the existing epoch-30 evaluation exactly: 0 count mismatches
over 12 runs x 1038 patches, and real counts equal the cache on 1038/1038.

## 1. Detector checks (all 1038 real patches; all 12,456 fakes)

| detector | (a) r vs StarDist | (a) slope | level vs SD | R1 rt/orig count | (c) empty fakes | (c) < 5 nuclei | (c) palest/darkest | (d) s/img, shared GPU (solo smoke) | detector gates |
|---|---|---|---|---|---|---|---|---|---|
| StarDist (ref.) | 1 | 1 | 1 | 1.000 | 0.0% | 0.1% | n/a | 0.16 (0.12) | PASS |
| CellViT-SAM-H | 0.975 | 0.99 | 1.20 | 0.966 | 0.0% | 0.2% | 1.10 | 0.20 (0.14) | PASS |
| InstanSeg | 0.968 | 1.04 | 1.00 | 1.030 | 0.0% | 0.1% | 1.07 | 0.03 (0.03) | PASS |
| Cellpose-SAM | 0.972 | 1.06 | 1.02 | 1.019 | 0.1% | 0.4% | 0.99 | 0.42 (0.33) | PASS (counts only) |
| HoVer-Net | 0.970 | 0.99 | 1.35 | **0.900** | 0.0% | 0.1% | 1.05 | 0.23 (0.15) | **FAIL (R1)** |

- R1 confirmed on the full set: the round trip costs HoVer-Net 10% of its nuclei (mean -7.6 per patch); StarDist
  loses none (median ratio 1.000, mean -0.2 nuclei), CellViT 3.4%. **The stain decomposition itself does not lose
  nuclei for StarDist**, so it is not a source of the fake undercount (handover open item 6b).
- No detector fails on pale fakes: second detectors find relatively *more* nuclei than StarDist in the palest third
  (`fig_paleness.png`).
- Timing was measured with 3-4 jobs sharing the GPU; every detector passes (d) either way. Stage B estimates per
  detector at shared-GPU speed: CellViT 2.9 h, InstanSeg 0.5 h, Cellpose 6.1 h.

## 2. ONE TABLE: every screened row (rt baseline; full CIs in `screen_table.csv`)

b = pooled within-model seed SD / SD of the four model means (pass <= 0.5). a(metric) = patch-level r with
StarDist on real (pass >= 0.70; StarDist is the reference).

| metric | detector | CG | CG+M | CG+M+A | CG+M+C | b | a(metric) | verdict |
|---|---|---|---|---|---|---|---|---|
| M6 density r | StarDist | 0.10 | 0.36 | 0.50 | 0.60 | 0.56 | ref | fail (b) |
| M6 density slope | StarDist | 0.11 | 0.26 | 0.35 | 0.44 | 0.56 | ref | fail (b) |
| M6 density disc | StarDist | 0.35 | 0.43 | 0.48 | 0.55 | **0.40** | ref | **PASS** |
| M6 density r | CellViT-SAM-H | 0.09 | 0.46 | 0.55 | 0.62 | 0.57 | | fail (b) |
| M6 density slope | CellViT-SAM-H | 0.10 | 0.32 | 0.41 | 0.43 | 0.66 | | fail (b) |
| M6 density disc | CellViT-SAM-H | 0.35 | 0.45 | 0.50 | 0.56 | 0.59 | | fail (b) |
| M6 density r | InstanSeg | 0.14 | 0.48 | 0.56 | 0.66 | 0.64 | | fail (b) |
| M6 density slope | InstanSeg | 0.11 | 0.30 | 0.41 | 0.43 | 0.66 | | fail (b) |
| M6 density disc | InstanSeg | 0.35 | 0.43 | 0.51 | 0.53 | 0.65 | | fail (b) |
| M6 density r | Cellpose-SAM | 0.08 | 0.41 | 0.50 | 0.60 | 0.54 | | fail (b) |
| M6 density slope | Cellpose-SAM | 0.08 | 0.28 | 0.37 | 0.43 | 0.65 | | fail (b) |
| M6 density disc | Cellpose-SAM | 0.33 | 0.43 | 0.50 | 0.56 | 0.55 | | fail (b) |
| M6 density r / slope | HoVer-Net | 0.08 / 0.09 | 0.38 / 0.27 | 0.53 / 0.38 | 0.59 / 0.38 | 0.55 / 0.60 | | fail (R1, b) |
| M6 density disc | HoVer-Net | 0.34 | 0.43 | 0.51 | 0.55 | 0.495 | | fail (R1 only) |
| M1 nuclear area | StarDist | 0.91 | 0.87 | 0.82 | 0.93 | 1.71 | ref | fail (b) |
| M1 nuclear area | CellViT-SAM-H | 0.95 | 0.91 | 0.81 | 0.96 | 1.54 | 0.91 | fail (b) |
| M1 nuclear area | HoVer-Net | 0.90 | 0.83 | 0.78 | 0.86 | 1.31 | 0.91 | fail (R1, b) |
| M2 solidity | StarDist | 1.00 | 1.00 | 1.00 | 1.00 | 1.32 | ref | fail (b) |
| M2 solidity | CellViT-SAM-H | 1.00 | 1.00 | 0.99 | 1.00 | 0.88 | 0.80 | fail (b) |
| M2 eccentricity | StarDist | 1.00 | 0.99 | 1.01 | 0.97 | 0.67 | ref | fail (b) |
| M2 eccentricity | CellViT-SAM-H | 0.99 | 0.98 | 1.00 | 0.96 | 0.67 | 0.87 | fail (b) |
| M3 nearest-neighbour dist. | StarDist | 1.03 | 1.09 | 1.10 | 1.11 | 1.96 | ref | fail (b) |
| M3 nearest-neighbour dist. | CellViT-SAM-H | 0.99 | 1.08 | 1.10 | 1.10 | 1.37 | 0.94 | fail (b) |
| M3 nearest-neighbour dist. | InstanSeg | 1.03 | 1.10 | 1.12 | 1.11 | 1.70 | 0.81 | fail (b) |
| M3 nearest-neighbour dist. | HoVer-Net | 0.94 | 1.02 | 1.04 | 1.03 | 1.37 | 0.80 | fail (R1, b) |
| M3 cell territory | StarDist | 1.02 | 1.14 | 1.20 | 1.23 | 2.26 | ref | fail (b) |
| M3 cell territory | CellViT-SAM-H | 0.93 | 1.12 | 1.21 | 1.20 | 1.77 | **0.65** | fail (a, b) |
| M3 cell territory | InstanSeg | 1.02 | 1.16 | 1.28 | 1.21 | 1.97 | 0.80 | fail (b) |
| M3 cell territory | HoVer-Net | 0.85 | 1.04 | 1.09 | 1.08 | 1.62 | 0.88 | fail (R1, b) |
| M4 nuclear H OD | StarDist | 1.11 | 1.14 | 1.11 | 1.04 | 2.12 | ref | fail (b) |
| M4 nuclear H OD | CellViT-SAM-H | 1.09 | 1.14 | 1.10 | 1.04 | 1.99 | 0.98 | fail (b) |
| M4 nuclear H OD | HoVer-Net | 1.08 | 1.14 | 1.09 | 1.04 | 1.74 | 0.97 | fail (R1, b) |
| M5 mean H in tissue | none | 1.10 | 1.14 | 0.96 | 1.02 | 0.93 | n/a | fail (b) |
| M5 mean E in tissue | none | 1.06 | 0.84 | 0.92 | 1.00 | 1.54 | n/a | fail (b) |
| M5 H fraction | none | 1.02 | 1.10 | 1.02 | 1.01 | 1.26 | n/a | fail (b) |

Not computed by eligibility (smoke verdicts): M2 for HoVer-Net, InstanSeg, Cellpose; M1/M4 for InstanSeg and
Cellpose; M3 for Cellpose.

**Pre-registered verdict: one row passes (StarDist M6 disc). Every other row fails criterion (b)**, except
HoVer-Net disc, which fails R1 only; all HoVer-Net rows fail R1, and CellViT territory also fails (a).

## 3. What the numbers say (interpretation, not criteria)

1. **Criterion (b) is strict at a single checkpoint; M6 is close, M1-M5 are not.** Post-hoc, from the existing eval
   CSVs (StarDist only, `posthoc_b_ratio_checkpoint_averaging.csv`): averaging epochs 27-30 cuts the StarDist b ratio
   from 0.57 to 0.48 (r), 0.57 to 0.40 (slope), 0.45 to 0.38 (disc). The second detectors' M6 rows sit at
   0.54-0.66 now, so Stage B could move some under 0.5. M1-M5 sit at 0.67-2.3 and would need a 2-4x reduction;
   averaging checkpoints cannot remove seed-to-seed variance, so **M1-M5 do not separate the models** beyond seed noise.
2. **M6 is detector-robust in order.** Under all five detectors the four models keep the same order on r, slope
   and disc (CG < CG+M < CG+M+A < CG+M+C), except that CG+M+A and CG+M+C nearly tie on slope under CellViT and
   InstanSeg (0.41 vs 0.43) and tie under HoVer-Net (0.38). Values agree with StarDist within 0.12; the largest
   gaps are CG+M (r 0.46-0.48 vs 0.36), i.e. second detectors see slightly better conditioning in CG+M. This is
   the most useful output of the screen for the paper.
3. **M1: fake nuclei are smaller in every model** (StarDist 0.82-0.93, CIs below 1; CellViT 0.81-0.96), smallest in
   CG+M+A (the area-loss model, 0.79-0.84 across density classes). A model-independent description that partly
   re-checks the old "smaller, not merged" finding on v2 models (handover open item 2; the "inverted area loss"
   part is not addressed here).
4. **M4 and M5: stain.** Fake nuclei are darker on average than real (1.04-1.14) but never exceed 0.125 OD, the
   calibration ceiling; real nuclei have a dark tail beyond it (`fig_nucleus_distributions.png`), so M4 partly
   measures the calibration range. In stromal patches all models put too much haematoxylin in tissue (mean H
   1.10-1.48 x real; H fraction 1.05-1.20), slightly too little in dense tissue (0.90-1.03): the stain
   counterpart of stromal fabrication and compression (`by_density_class.csv`).
5. **M2 shape** carries almost no signal (solidity 0.99-1.00; eccentricity 0.96-1.01).
6. **M3 spacing** moves with count (territory is close to tissue area / count) and adds little beyond M6.

## 4. Ranked recommendation (for Stage B and the paper; my judgement)

Detectors:
1. **StarDist** stays the reference (reproduces every existing number).
2. **CellViT-SAM-H**: passes every detector gate, cleanest outlines, the only second detector eligible for shape.
   Recommended second detector.
3. **InstanSeg**: passes every gate, matches StarDist's level (1.00), 0.03 s/image. Recommended as a cheap third
   detector for M6 only.
4. Cellpose-SAM: passes the count gates but outlines whole cells and is the slowest; adds nothing over InstanSeg. Drop.
5. HoVer-Net: fails R1 (10% round-trip loss) and fragments dense nuclei. Drop.

Metrics:
1. **M6 under CellViT-SAM-H and InstanSeg** as a detector-robustness check of the main result. Stage B (epochs
   27-30) is justified for these two only; estimated about 3.5-4 h (generation about 45 min, CellViT about 2.9 h,
   InstanSeg about 0.5 h; StarDist re-run not needed for counts, as the existing CSVs hold them, but needed if
   per-nucleus tables for epochs 27-29 are wanted, about 2.3 h on CPU in parallel).
2. **R1** result as a methods statement (round trip loses no StarDist nuclei).
3. **M1 and M5 by density class** as descriptive, all-model statements only (they fail (b), so they must not be
   used to rank models).
4. Not recommended: M2, M3, M4 (no signal, redundant with count, or bounded by the calibration).

## Files
`outputs\s18_metric_screen\stageA\`: screen_table.csv, per_run_values.csv, by_density_class.csv,
detector_checks.csv, R1_roundtrip_metrics.csv, sanity_stardist_vs_eval.csv,
posthoc_b_ratio_checkpoint_averaging.csv, fig_agreement_real.png, fig_run_values.png, fig_paleness.png,
fig_nucleus_distributions.png. Bulk (D:\vhe_count_work\s18_metric_screen\stageA\): tables\<detector>\<set>_nuclei.csv.gz
(per-nucleus: centroid, area, solidity, eccentricity, mean H, mean E) and <set>_patches.csv for 5 detectors x 14
image sets (0.13 GB); gallery\<set>\ (40 stratified patches x 14 sets, 560 PNGs); patches.csv. Scripts s18lib.py,
s18f-s18j, run_s18_stageA.ps1.
