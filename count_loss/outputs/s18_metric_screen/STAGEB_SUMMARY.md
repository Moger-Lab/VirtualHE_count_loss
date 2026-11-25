# s18 metric/detector screen: Stage B summary (2026-10-02)

Rules fixed beforehand: DECISION_LOG D-31. Scope: M6 (density r, log-log slope, disc) for StarDist (reference, existing eval CSVs), CellViT-SAM-H and InstanSeg; per-patch fake count averaged over epochs 27-30; 12 runs (4 models x 3 seeds); pooled held-out set (17 cores, 1038 patches). Baseline: real round trip; the original-RGB baseline is in the CSV.
Run: epochs 27-29 generated and detected (37,368 images per detector; epoch 30 reused from Stage A). StarDist from the eval CSVs equals Stage A at epoch 30 (0 mismatches).

## Detector checks on the new epochs
| detector | empty fakes | <5 nuclei | pale/dark count ratio | median s/img | (c) | (d) |
|---|---|---|---|---|---|---|
| CellViT-SAM-H | 0.0% | 0.2% | 1.06 | 0.145 | pass | pass |
| InstanSeg | 0.0% | 0.2% | 1.05 | 0.030 | pass | pass |
(a) and R1 carry over from Stage A (both pass).

## All Stage B rows (mean of epochs 27-30; b = seed SD / SD of the four model means; pass needs b <= 0.5)
| metric | detector | CG | CG+M | CG+M+A | CG+M+C | b, epoch 30 | b, epochs 27-30 | verdict |
|---|---|---|---|---|---|---|---|---|
| M6 r | StarDist | 0.12 | 0.41 | 0.55 | 0.64 | 0.56 | 0.47 | PASS |
| M6 slope | StarDist | 0.11 | 0.27 | 0.38 | 0.47 | 0.56 | 0.38 | PASS |
| M6 disc | StarDist | 0.36 | 0.44 | 0.49 | 0.58 | 0.40 | 0.39 | PASS |
| M6 r | CellViT-SAM-H | 0.11 | 0.50 | 0.59 | 0.66 | 0.57 | 0.59 | fail (b) |
| M6 slope | CellViT-SAM-H | 0.09 | 0.33 | 0.43 | 0.46 | 0.66 | 0.53 | fail (b) |
| M6 disc | CellViT-SAM-H | 0.35 | 0.46 | 0.53 | 0.57 | 0.59 | 0.51 | fail (b), narrowly |
| M6 r | InstanSeg | 0.15 | 0.52 | 0.61 | 0.68 | 0.64 | 0.64 | fail (b) |
| M6 slope | InstanSeg | 0.10 | 0.32 | 0.43 | 0.44 | 0.66 | 0.52 | fail (b), narrowly |
| M6 disc | InstanSeg | 0.35 | 0.45 | 0.52 | 0.54 | 0.65 | 0.55 | fail (b) |
Bootstrap CIs (cores resampled jointly across seeds) are in stageB_table.csv. Checkpoint-to-checkpoint SD within a run: 0.03-0.06, so averaging four checkpoints removes part of the noise but not the seed-to-seed part.

## What it means
- With checkpoint averaging StarDist passes all three M6 rows (b 0.38-0.47). CellViT-SAM-H and InstanSeg improve on slope and disc (b 0.51-0.55, just above 0.5) but not on r. By the pre-registered rule they fail, so StarDist is the only detector-metric combination that passes (M6 r, slope, disc).
- Most of the seed variance sits in CG: its three seeds span r about -0.2 to 0.3 under every detector, while CG+M, CG+M+A and CG+M+C seeds agree to about 0.05. The b ratio is therefore driven by one unstable model, not by detector disagreement.
- All three detectors give the same model order on every M6 metric (CG < CG+M < CG+M+A < CG+M+C); the CG+M+A and CG+M+C bootstrap CIs overlap on slope under CellViT and InstanSeg.
- Cost: both second detectors are cheap enough to add as a robustness panel (CellViT 0.145 s/img, InstanSeg 0.03 s/img).

Files: outputs\s18_metric_screen\stageB\ (stageB_table.csv, per_run_values.csv, detector_checks_stageB.csv, fig_stageB_run_values.png); per-nucleus and per-patch tables in D:\vhe_count_work\s18_metric_screen\stageB_e27|28|29\tables\ (122 MB); copied to the Drive lock-down folder.
