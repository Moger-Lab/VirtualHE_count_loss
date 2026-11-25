# Density-conditioned virtual H&E from label-free SRS: a local nucleus-count loss

Code, released per-patch tables and figure sources for:

> [INSERT paper DOI when accepted]

We translate stimulated Raman scattering (SRS) images of breast tissue-microarray cores into virtual
H&E, training against the H&E-stained **adjacent** section (no cell-level correspondence). Four
models are trained identically (three seeds each): **CG** (CycleGAN), **CG+M** (paired-location
sampling and a macro-architecture loss), and two alternatives branching from CG+M: **CG+M+A**
(nuclear-area loss) and **CG+M+C** (local nucleus-count loss: a frozen counter on the generated
haematoxylin channel, 2 x 2 cells of 50 um per patch). Evaluation counts nuclei with StarDist in
the generated and real images of 17 held-out cores (1038 patches of 100 um).

## Main results

Held-out evaluation cores (17 cores, 1038 patches), StarDist, epochs 27-30, mean over 3 seeds
[95% core-cluster bootstrap CI]. Slope = log-log slope of generated on real nucleus count
(1 = proportional); disc = dense/sparse generated-to-real ratio (1 = no compression).

| model | per-patch r | log-log slope | disc |
|---|---|---|---|
| CG | 0.12 [-0.02, 0.28] | 0.11 [0.00, 0.25] | 0.36 [0.29, 0.46] |
| CG+M | 0.41 [0.31, 0.53] | 0.27 [0.19, 0.37] | 0.44 [0.36, 0.51] |
| CG+M+A | 0.55 [0.45, 0.65] | 0.38 [0.27, 0.50] | 0.49 [0.42, 0.58] |
| CG+M+C | 0.64 [0.56, 0.73] | 0.47 [0.35, 0.61] | 0.58 [0.48, 0.66] |

Pre-specified contrast CG+M+C minus CG+M+A: r +0.09 [0.04, 0.14], slope +0.09 [0.04, 0.14],
disc +0.09 [0.04, 0.13] (see `docs/PRESPECIFICATION.md`). The gain in correlation is
detector-robust; the slope and dense/sparse gains are seen with StarDist and are smaller with
independent detectors (CellViT-SAM-H, InstanSeg; Supplementary Fig. S5).

Reproduce this table and the data figures from the released tables (CPU only, about 15 minutes):

```
python reproduce_tables_and_figures.py --out regenerated
```

It regenerates the statistics and Figs 3, 4, 5, S2, S3, S4, S5, S6a, S7, S8 into `regenerated/` and
compares every table with the released copy; on the authors' machine all 34 tables were identical.

## Reproducibility levels

| level | what you can reproduce | needs |
|---|---|---|
| 1 | every statistic and the data figures | this repository only (CPU) |
| 2 | inference, StarDist evaluation, detector screen, example-image figures (Figs 1, 2, S1, S6b, S9) | the registered image stacks and the trained weights (see Data and weights) |
| 3 | everything from raw images: registration, counter, training (12 runs) | the raw images; one GPU with >= 16 GB (about 2.5 h per 30-epoch run on an RTX 5070 Ti) |

## Requirements

Three conda environments (exported files in `environment/`, versions as used):

- `srs-analysis` (Python 3.11, PyTorch 2.11 + CUDA 12.8, StarDist 0.9.2 with TensorFlow on CPU): everything except the second detectors.
- `vhe-detectors` (InstanSeg, Cellpose-SAM, HoVer-Net via TIAToolbox): detector screen only.
- `vhe-cellvit` (CellViT 1.0.9; separate because of its numpy pin): detector screen only.

```
conda env create -f environment/srs-analysis.yml
conda activate srs-analysis
```

Level 1 needs only numpy, pandas, scipy, scikit-image, matplotlib and Pillow from `srs-analysis`.
Developed and run on Windows 11; scripts are plain Python and path-independent (see below).

## Paths

All machine-specific locations are in `count_loss/vhe_paths.py` and can be set with environment
variables; the defaults point inside the repository.

| variable | default | contents |
|---|---|---|
| `VHE_COUNT_WORK` | `work/` | evaluation CSVs, detector tables, counter weights, checkpoints (`work/runs/<run>/`) |
| `VHE_ALIGNED_DIR` | `data/aligned/` | registered SRS/H&E stacks (`<core>_srs.npy`, `<core>_he.npy`, `<core>_valid.npy`) |
| `VHE_RAW_DATA` | `data/raw/` | raw images: `fsSRS/<core>_fSRS.tif`, `H&E/<core>_H&E.png` |
| `VHE_V2_DIR` | `registration/` | registration code and its result tables |
| `VHE_FIG_OUT` | the figure folder | where figure scripts write (used by the reproduction script) |

Only path constants differ from the code that produced the paper; every change is listed in
`docs/PORTABILITY_CHANGES.md`.

## Data and weights

- **Images.** Raw SRS/SHG/CARS/TPEF images and adjacent-section H&E scans of 77 TMA cores are not
  included. [OWNER: availability statement, e.g. "available from the corresponding author on
  reasonable request" or a repository DOI.]
- **Trained weights.** [OWNER: Zenodo DOI] Generator checkpoints `G_S2H_epoch027-030.pt` of the 12
  runs (4 models x 3 seeds; about 45 MB each). Place them in `work/runs/<run>/`. Run names:
  `CG_full`, `CGM_full`, `A1_full` (CG+M+A), `A5_full` (CG+M+C), with suffix `_s7` or `_s123` for
  seeds 7 and 123 (no suffix: seed 42).
- **Included here:** the frozen counter (`work/counter/counter_best.pt`, 3 MB); StarDist results on
  the real patches (`work/stardist_v2/`); per-core SRS normalisation (`work/srs_norm/`); the stain
  calibration (`count_loss/vhe_v1/calibration/`); per-patch evaluation tables of every evaluated
  checkpoint (`work/runs/`); per-patch detector tables (`work/s18_metric_screen/`); registration
  results (`registration/results/`); training configurations and loss histories
  (`count_loss/outputs/runs/`); every figure's source data.

## Pipeline (levels 2 and 3)

Run from `registration/` or `count_loss/` as indicated. Internal run names: A1 = CG+M+A, A5 = CG+M+C.

**1. Registration** (`registration/`, raw images in `VHE_RAW_DATA`): scripts `00_inventory.py`
to `16_export_transforms.py` in numerical order (core segmentation, scale, orientation,
registration and QC, scan coverage, nuclear density, cohort tiers, stain quality, aligned stacks,
cross-validation design, patch manifest, transforms). `scripts/regenerate_aligned.py` rebuilds the
`<core>_srs.npy` and `<core>_he.npy` stacks from the raw images and `results/16_transforms.csv`
alone (the `<core>_valid.npy` masks come from `13_build_aligned_stacks.py`). SHA-256 checksums of
the 225 aligned arrays used in the paper: `docs/aligned_sha256.csv`.

**2. Caches and the counter** (`count_loss/`):

```
python s02a_stardist_cache.py            # StarDist on real patches (shardable: --shard i --nshards n, then --merge)
python s02b_srs_norm_cache.py            # per-core SRS percentiles
python s11_cars_thresholds.py            # per-core background thresholds
python s02c_train_counter.py             # frozen counter (training folds only)
python s02d_counter_acceptance.py        # counter validation (Supplementary Fig. S3)
python s02e_acceptance_diagnostics.py
python test_step4_units.py               # unit tests of the losses and loaders
```

**3. Training** (12 runs; shown for seed 42, repeat with `--seed 7 --tag <name>_s7` and `--seed 123 --tag <name>_s123`):

```
python s03b_train_cg.py   --arm CG  --epochs 30 --seed 42 --tag CG_full  --samples_per_epoch 6076 --ckpt_every 1
python s03b_train_cg.py   --arm CGM --epochs 30 --seed 42 --tag CGM_full --samples_per_epoch 6076 --ckpt_every 1
python s03_train_count.py --arm A1  --subset full --epochs 30 --seed 42 --tag A1_full --deterministic 0 --samples_per_epoch 6076 --ckpt_every 1
python s03_train_count.py --arm A5  --subset full --epochs 30 --seed 42 --tag A5_full --deterministic 0 --samples_per_epoch 6076 --ckpt_every 1
```

The recorded configuration of every run is in `count_loss/outputs/runs/<run>/config.json`.
Training is not bit-deterministic (cuDNN); expect run-to-run differences of the size of the seed
spread.

**4. Evaluation** (StarDist, epochs 27-30, both held-out sets):

```
python s05_eval_arms.py --tags <run> --epochs 27 28 29 30 --split val --eval_only
python s05_eval_arms.py --tags <run> --epochs 27 28 29 30 --split legacy_test --eval_only
```

(`legacy_test` is the internal name of the previously held-out cores; together with the validation
fold they form the 17 held-out evaluation cores.)

**5. Detector screen** (Supplementary Fig. S5): `s18f_stageA_images.py` (generated, real and
round-trip images), `s18g_stageA_detect.py --det <detector>` in the matching environment,
`s18i_stageA_report.py`; epochs 27-29: `s18f_stageA_images.py --epoch <e> --out stageB_e<e>`, then
`s18g` with `--root`, then `s18k_stageB_report.py`. The PowerShell launchers `run_s18_stage*.ps1`
run the streams in parallel (set `VHE_CONDA_ENVS`).

**6. Statistics and figures** (`count_loss/paper_figures/final_2026-10/`):

```
python a01_tables.py          # four-model table, per-run values, paired differences
python a02_descriptive.py     # nuclear size and stain by density class; checkpoint fabrication
python a03_area_signal.py     # area-loss signal of CG+M+A (GPU inference; Supplementary Fig. S8a)
python a04_detector_paired.py # paired differences under each detector
python a05_loss_shares.py     # loss share by cell size (Supplementary Fig. S8b)
python g01_candidate_images.py  # example-patch images and rule-based selection
python fig1_schematic.py, fig2_candidates.py, fig3_four_models.py, fig4_patch_level.py, fig5_what_remains.py,
       figS1_registration_examples.py, figS2_registration.py, ..., figS8_design.py, final_selected.py
```

Analysis scripts refuse to overwrite existing outputs; set `VHE_FIG_OUT` to an empty folder (as the
reproduction script does). `FIGURE_LEGENDS_2026-10.md` holds the legends.

## Repository layout

```
reproduce_tables_and_figures.py   level-1 reproduction and comparison
registration/                     registration of the adjacent H&E onto the SRS grid; results/
count_loss/                       datasets, losses, counter, training, evaluation, detector screen
  vhe_paths.py                    all machine-specific paths
  vhe_v1/                         helper modules (stain conversion, ResNet generator, PatchGAN) and stain calibration
  outputs/                        run configurations, counter validation, detector-screen summaries
  paper_figures/final_2026-10/    statistics and figure scripts, data/, figures (PDF, PNG), source CSVs, legends
work/                             released per-patch tables and small weights (layout of the original work folder)
environment/                      conda environments
docs/                             PRESPECIFICATION.md, PORTABILITY_CHANGES.md
```

## Citation

See `CITATION.cff`. [OWNER: add the paper reference and the Zenodo DOI of this release.]

## Licence

Code: see `LICENSE`. Third-party models used (not redistributed): StarDist (BSD-3-Clause),
CellViT ("Apache 2.0 with Commons Clause", non-commercial), InstanSeg (Apache-2.0).
