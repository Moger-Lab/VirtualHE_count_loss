# s18 Stage B: epochs 27-29 for CellViT-SAM-H and InstanSeg (epoch 30 reused from Stage A).
# Stream gen: generate fakes per epoch and drop a GEN_DONE marker. Streams cv / is: wait for
# the marker, then detect the 12 run sets. Resumable (s18f and s18g skip finished work).
$proj = $PSScriptRoot
$envs = $env:VHE_CONDA_ENVS  # folder holding the conda envs srs-analysis, vhe-detectors, vhe-cellvit
$work = Join-Path $(if ($env:VHE_COUNT_WORK) { $env:VHE_COUNT_WORK } else { Join-Path $PSScriptRoot "..\work" }) "s18_metric_screen"
$epochs = 27, 28, 29
$tags = "CG_full CG_full_s7 CG_full_s123 CGM_full CGM_full_s7 CGM_full_s123 A1_full A1_full_s7 A1_full_s123 A5_full A5_full_s7 A5_full_s123"
$pre = "`$env:PYTHONIOENCODING='utf-8'; Set-Location '$proj'; "

$gen = foreach ($e in $epochs) {
    "& '$envs\srs-analysis\python.exe' s18f_stageA_images.py --epoch $e --out stageB_e$e *>> '$proj\logs\s18_stageB_gen.log'; " +
    "if ((Get-ChildItem '$work\stageB_e$e\img\A5_full_s123' -Filter *.png).Count -eq 1038) { 'ok' | Out-File '$work\stageB_e$e\GEN_DONE' }"
}
$streams = @(
    @{ name = "gen"; body = ($gen -join "; ") },
    @{ name = "cv"; body = ((foreach ($e in $epochs) {
        "while (-not (Test-Path '$work\stageB_e$e\GEN_DONE')) { Start-Sleep 30 }; " +
        "& '$envs\vhe-cellvit\python.exe' s18g_stageA_detect.py --det cellvit_sam --root '$work\stageB_e$e' --sets $tags *>> '$proj\logs\s18_stageB_cv.log'" }) -join "; ") },
    @{ name = "is"; body = ((foreach ($e in $epochs) {
        "while (-not (Test-Path '$work\stageB_e$e\GEN_DONE')) { Start-Sleep 30 }; " +
        "& '$envs\vhe-detectors\python.exe' s18g_stageA_detect.py --det instanseg --root '$work\stageB_e$e' --sets $tags *>> '$proj\logs\s18_stageB_is.log'" }) -join "; ") }
)
foreach ($s in $streams) {
    $log = "$proj\logs\s18_stageB_$($s.name).log"
    $script = $pre + $s.body + "; 'STREAM DONE' | Out-File -Append '$log'"
    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $script -WindowStyle Hidden
    Write-Output "started stream $($s.name) -> $log"
}
