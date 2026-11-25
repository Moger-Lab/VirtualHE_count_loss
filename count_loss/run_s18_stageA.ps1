# s18 Stage A detector queue: three parallel streams (StarDist on CPU; two GPU streams).
# Resumable: s18g skips finished image sets. Logs in logs\s18_stageA_<stream>.log
$proj = $PSScriptRoot
$envs = $env:VHE_CONDA_ENVS  # folder holding the conda envs srs-analysis, vhe-detectors, vhe-cellvit
$streams = @(
    @{ name = "cpu";  cmds = @(, @("srs-analysis", "stardist")) },  # leading comma: keep the nested array
    @{ name = "gpu1"; cmds = @(@("vhe-cellvit", "cellvit_sam"), @("vhe-detectors", "cellpose")) },
    @{ name = "gpu2"; cmds = @(@("vhe-detectors", "hovernet"), @("vhe-detectors", "instanseg")) }
)
foreach ($s in $streams) {
    $log = Join-Path $proj "logs\s18_stageA_$($s.name).log"
    $lines = foreach ($c in $s.cmds) {
        "& '$envs\$($c[0])\python.exe' s18g_stageA_detect.py --det $($c[1]) *>> '$log'"
    }
    $script = "`$env:PYTHONIOENCODING='utf-8'; Set-Location '$proj'; " + ($lines -join "; ") + "; 'STREAM DONE' | Out-File -Append '$log'"
    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $script -WindowStyle Hidden
    Write-Output "started stream $($s.name) -> $log"
}
