# Changes made to the copied code

Only machine-specific paths were changed; no analysis logic. With the VHE_* environment variables pointing at the original folders, every script behaves as it did when the results were produced. Lines removed (-) and added (+), per file:

## count_loss/countlib.py

```
+import vhe_paths  # repository: all machine-specific locations live in vhe_paths.py
-V1_DIR = os.path.join(PY_SCRIPTS, "VirtualH&E_semi-VirtualH&E_semi-supervised_cycleGAN_30_08_26_RTX4500")
-V1_NONAUG_DIR = os.path.join(PY_SCRIPTS, "VirtualH&E_semi-VirtualH&E_semi-supervised_cycleGAN_30_08_26")
-V2_DIR = os.path.join(PY_SCRIPTS, "VirtualH&E_Analysis_v2")
+V1_DIR = os.path.join(_this_dir, "vhe_v1")  # bundled copy of the helper modules (common.py, models.py, calibration)
+V1_NONAUG_DIR = V1_DIR  # only used by historical diagnostics not part of the paper
+V2_DIR = vhe_paths.V2_DIR
-ALIGNED_DIR = os.environ.get("VHE_ALIGNED_DIR", r"D:\vhe_v2_work\aligned")
-WORK_DIR = os.environ.get("VHE_COUNT_WORK", r"D:\vhe_count_work")
+ALIGNED_DIR = vhe_paths.ALIGNED
+WORK_DIR = vhe_paths.WORK
```

## count_loss/run_cg_queue.ps1

```
-$py = "C:\Users\<local user>\miniforge3\envs\srs-analysis\python.exe"
+$py = if ($env:VHE_PYTHON) { $env:VHE_PYTHON } else { "python" }
+$work = if ($env:VHE_COUNT_WORK) { $env:VHE_COUNT_WORK } else { Join-Path $PSScriptRoot "..\work" }
-    if (Test-Path -LiteralPath "D:\vhe_count_work\runs\$tag\G_S2H_epoch030.pt") { continue }
+    if (Test-Path -LiteralPath "$work\runs\$tag\G_S2H_epoch030.pt") { continue }
-    if (Test-Path -LiteralPath "D:\vhe_count_work\runs\$tag\latest.pt") { $resume = "--resume" }
+    if (Test-Path -LiteralPath "$work\runs\$tag\latest.pt") { $resume = "--resume" }
```

## count_loss/run_s18_stageA.ps1

```
-$proj = "C:\Users\<local user>\OneDrive - University of Exeter\Python_Scripts\VirtualH&E_count_loss"
-$envs = "C:\Users\<local user>\miniforge3\envs"
+$proj = $PSScriptRoot
+$envs = $env:VHE_CONDA_ENVS  # folder holding the conda envs srs-analysis, vhe-detectors, vhe-cellvit
```

## count_loss/run_s18_stageB.ps1

```
-$proj = "C:\Users\<local user>\OneDrive - University of Exeter\Python_Scripts\VirtualH&E_count_loss"
-$envs = "C:\Users\<local user>\miniforge3\envs"
-$work = "D:\vhe_count_work\s18_metric_screen"
+$proj = $PSScriptRoot
+$envs = $env:VHE_CONDA_ENVS  # folder holding the conda envs srs-analysis, vhe-detectors, vhe-cellvit
+$work = Join-Path $(if ($env:VHE_COUNT_WORK) { $env:VHE_COUNT_WORK } else { Join-Path $PSScriptRoot "..\work" }) "s18_metric_screen"
```

## count_loss/s18d_smoke_report.py

```
-IN_DIR = r"D:\vhe_count_work\s18_metric_screen\smoke"
+IN_DIR = __import__("vhe_paths").work("s18_metric_screen", "smoke")
```

## count_loss/s18e_examples.py

```
-IN_DIR = r"D:\vhe_count_work\s18_metric_screen\smoke"
+IN_DIR = __import__("vhe_paths").work("s18_metric_screen", "smoke")
```

## count_loss/s18g_stageA_detect.py

```
-DEFAULT_ROOT = r"D:\vhe_count_work\s18_metric_screen\stageA"
+DEFAULT_ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageA")
```

## count_loss/s18i_stageA_report.py

```
-ROOT = os.environ.get("S18_ROOT", r"D:\vhe_count_work\s18_metric_screen\stageA")
-WORK_RUNS = r"D:\vhe_count_work\runs"
+ROOT = os.environ.get("S18_ROOT", __import__("vhe_paths").work("s18_metric_screen", "stageA"))
+WORK_RUNS = __import__("vhe_paths").work("runs")
```

## count_loss/s18j_stageA_cleanup.py

```
-ROOT = r"D:\vhe_count_work\s18_metric_screen\stageA"
+ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageA")
```

## count_loss/s18k_stageB_report.py

```
-A_ROOT = r"D:\vhe_count_work\s18_metric_screen\stageA"
-B_ROOT = r"D:\vhe_count_work\s18_metric_screen\stageB_e{e}"
-RUNS_DIR = r"D:\vhe_count_work\runs"
+A_ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageA")
+B_ROOT = __import__("vhe_paths").work("s18_metric_screen", "stageB_e{e}")
+RUNS_DIR = __import__("vhe_paths").work("runs")
```

## count_loss/paper_figures/final_2026-10/a04_detector_paired.py

```
-STAGEB = r"D:\vhe_count_work\s18_metric_screen\stageB_e{e}"
+STAGEB = __import__("vhe_paths").work("s18_metric_screen", "stageB_e{e}")
```

## count_loss/paper_figures/final_2026-10/fcommon.py

```
-OUT = _here
+OUT = os.environ.get("VHE_FIG_OUT", _here)  # repository: write regenerated outputs elsewhere
-STAGEA = r"D:\vhe_count_work\s18_metric_screen\stageA"
-RUNS_DIR = r"D:\vhe_count_work\runs"
+STAGEA = __import__("vhe_paths").work("s18_metric_screen", "stageA")
+RUNS_DIR = __import__("vhe_paths").work("runs")
```

## count_loss/paper_figures/final_2026-10/fig1_schematic.py

```
-IMG = r"D:\vhe_count_work\paper_figs_2026-10\img"
+IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
```

## count_loss/paper_figures/final_2026-10/fig2_candidates.py

```
-IMG = r"D:\vhe_count_work\paper_figs_2026-10\img"
+IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
```

## count_loss/paper_figures/final_2026-10/figS1_registration_examples.py

```
-RAW_HE = os.path.join(os.path.expanduser("~"), "OneDrive - University of Exeter", "DATA_ARCHIVE", "H&E_SRS",
-                      "Paired Images", "H&E")
+RAW_HE = os.path.join(__import__("vhe_paths").RAW_DATA, "H&E")
```

## count_loss/paper_figures/final_2026-10/g01_candidate_images.py

```
-IMG = r"D:\vhe_count_work\paper_figs_2026-10\img"
+IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")
```

## registration/config.py

```
-if not USERPROFILE.exists():  # pragma: no cover - guards the documented trap
-    raise RuntimeError(f"USERPROFILE does not exist: {USERPROFILE}")
+# repository: the original host check is disabled; set the VHE_* variables instead
-DATA_ROOT = ONEDRIVE / "DATA_ARCHIVE" / "H&E_SRS" / "Paired Images"
+DATA_ROOT = Path(os.environ.get("VHE_RAW_DATA", Path(__file__).resolve().parent.parent / "data" / "raw"))
```

## registration/config.py (second edit)

```n-USERPROFILE = Path(os.environ.get("USERPROFILE", r"C:\Users\<local user>"))
+USERPROFILE = Path(os.environ.get("USERPROFILE", str(Path.home())))
```

Also removed: registration/scripts/17_build_transfer_package.py (a local file-transfer helper, not part of the analysis).


## count_loss/paper_figures/final_2026-10/fcommon.py (second edit)

```n-DATA = os.path.join(_here, "data")
+DATA = os.path.join(OUT, "data")  # repository: follows VHE_FIG_OUT (default unchanged)
```


(Added with Fig 4 panel d)

## count_loss/paper_figures/final_2026-10/g02_core_nuclei.py
```
-IMGC = r"D:\vhe_count_work\paper_figs_2026-10\img_cores"
+IMGC = __import__("vhe_paths").work("paper_figs_2026-10", "img_cores")
```

## count_loss/paper_figures/final_2026-10/fig4_patch_level.py
```
-IMGC = r"D:\vhe_count_work\paper_figs_2026-10\img_cores"
+IMGC = __import__("vhe_paths").work("paper_figs_2026-10", "img_cores")
```

## count_loss/paper_figures/final_2026-10/core_panels.py
```
-IMGC = r"D:\vhe_count_work\paper_figs_2026-10\img_cores"
+IMGC = __import__("vhe_paths").work("paper_figs_2026-10", "img_cores")
```
(fig4_patch_level.py now takes this path from core_panels.CP.IMGC; Fig 4d and figS10_typical_core.py are skipped when the core mosaics are absent.)

(Round-core update: core_panels.py and g02_core_nuclei.py carry the same IMGC change as above; g03_core_full.py imports it from g02. Fig 4d, S10 and g03 need the core mosaics/arrays and are skipped or unavailable without them.)

## count_loss/paper_figures/final_2026-10/example_panels.py, g04_example_labels.py (new, visual example panels)

-IMG = r"D:\vhe_count_work\paper_figs_2026-10\img"
+IMG = __import__("vhe_paths").work("paper_figs_2026-10", "img")

Fig 3d and Fig 5d-f (example_panels.py) need the image folder paper_figs_2026-10\img (rendered patches), and Fig 4d / S10 need img_cores; neither is distributed, so those panels are skipped by the scripts when the folders are absent (panels a-c of Figs 3-5 are unaffected). The example nucleus labels used by the panels are included in count_loss/paper_figures/final_2026-10/data/example_labels, with the checks in data/fig_examples_checks.csv.
