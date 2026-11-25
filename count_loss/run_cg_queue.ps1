# Sequential queue: v2 CG and CG+M, seeds 42 / 7 / 123, 30 epochs, matched to A1 (DECISION_LOG D-24).
# Re-runnable: finished runs are skipped, interrupted runs resume from latest.pt.
$py = if ($env:VHE_PYTHON) { $env:VHE_PYTHON } else { "python" }
$work = if ($env:VHE_COUNT_WORK) { $env:VHE_COUNT_WORK } else { Join-Path $PSScriptRoot "..\work" }
Set-Location $PSScriptRoot
$jobs = @(@("CGM",42,"CGM_full"), @("CG",42,"CG_full"), @("CGM",7,"CGM_full_s7"), @("CG",7,"CG_full_s7"), @("CGM",123,"CGM_full_s123"), @("CG",123,"CG_full_s123"))
foreach ($j in $jobs) {
    $arm = $j[0]; $seed = $j[1]; $tag = $j[2]
    if (Test-Path -LiteralPath "$work\runs\$tag\G_S2H_epoch030.pt") { continue }
    $resume = ""
    if (Test-Path -LiteralPath "$work\runs\$tag\latest.pt") { $resume = "--resume" }
    New-Item -ItemType Directory -Force "outputs\runs\$tag" | Out-Null
    cmd /c "`"$py`" s03b_train_cg.py --arm $arm --epochs 30 --seed $seed --tag $tag --samples_per_epoch 6076 --ckpt_every 1 $resume >> outputs\runs\$tag\train.log 2>&1"
}
