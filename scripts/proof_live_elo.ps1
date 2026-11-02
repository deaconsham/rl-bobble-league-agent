# Live-Elo deadlock proof (Phase 12 review item 2).
#
# Runs a SHORT real training burst with the spare live-Elo game ON
# (--live-elo-shm), crossing several live-Elo boundaries, and checks the run
# completes (exit 0, no hang) with at least 3 "Live Elo @" checks logged. The
# training loop now OWNS the spare game's launch (and relaunch) via a dedicated
# one-game launcher, so --launch is all the caller needs.
#
# usage:
#   powershell -ExecutionPolicy Bypass -File scripts\proof_live_elo.ps1
#
# This is the review item's "prove it with a live run that crosses at least 3
# live-Elo boundaries with --live-elo-shm on". --live-elo-freq is lowered so a few
# epochs cross several boundaries; the handshake path exercised is identical.

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 6,
    [int]$TimeScale = 48,
    [int]$Epochs = 7,
    [int]$StepsPerEpoch = 600,
    [int]$LiveEloFreq = 2,
    [int]$LiveEloGames = 4
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$logDir = Join-Path $model "logs\proof_live_elo"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Fail($msg) { Write-Host "LIVE-ELO PROOF FAIL: $msg" -ForegroundColor Red; exit 1 }

$trainLog = Join-Path $logDir "train.log"
$shmPrefix = Join-Path $env:TEMP "proof_live_elo"
New-Item -ItemType Directory -Force -Path $shmPrefix | Out-Null
# the spare live-Elo game's shm (a KNOWN path so the loop's launcher can relaunch it)
$liveShm = Join-Path $shmPrefix "live_elo_spare.dat"
if (Test-Path $liveShm) { Remove-Item $liveShm -Force }

Write-Host "== train $Epochs x $StepsPerEpoch over $Games games, live-Elo every $LiveEloFreq epochs =="
$sw = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $model
& python training_loop.py --games $Games --steps-per-epoch $StepsPerEpoch `
    --epochs $Epochs --opponent chaser --launch --godot $GodotExe `
    --project $project --shm-prefix $shmPrefix --timescale $TimeScale `
    --live-elo-shm $liveShm --live-elo-freq $LiveEloFreq --live-elo-games $LiveEloGames `
    --best-path (Join-Path $logDir "best.pt") `
    --log-dir runs/proof_live_elo 2>&1 | Tee-Object -FilePath $trainLog
$trainCode = $LASTEXITCODE
Pop-Location
$sw.Stop()

$trainText = Get-Content $trainLog -Raw
$liveChecks = ([regex]::Matches($trainText, "Live Elo @ \d+:")).Count
$lastEpoch = ([regex]::Matches($trainText, "Epoch: (\d+)") | ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum
$pyErr = ([regex]::Matches($trainText, "Traceback \(most recent call last\)")).Count
$relaunches = ([regex]::Matches($trainText, "spare game relaunched")).Count

Write-Host ""
Write-Host "== live-Elo proof results ==" -ForegroundColor Cyan
Write-Host ("  run exit code:        {0}" -f $trainCode)
Write-Host ("  wall clock:           {0:N2} min" -f $sw.Elapsed.TotalMinutes)
Write-Host ("  last epoch reached:   {0} of {1}" -f $lastEpoch, ($Epochs - 1))
Write-Host ("  live-Elo checks run:  {0}" -f $liveChecks)
Write-Host ("  spare relaunches:     {0}" -f $relaunches)
Write-Host ("  python tracebacks:    {0}" -f $pyErr)

if ($trainCode -ne 0) { Fail "training exited $trainCode (a hang or crash)" }
if ($pyErr -gt 0) { Fail "python error in the run" }
if ($lastEpoch -lt ($Epochs - 1)) { Fail "run did not reach the final epoch (possible hang)" }
if ($liveChecks -lt 3) { Fail "only $liveChecks live-Elo checks crossed; need >= 3" }

Write-Host ""
Write-Host ("LIVE-ELO PROOF PASS: {0} live-Elo boundaries crossed, run finished epoch {1} exit 0" -f $liveChecks, $lastEpoch) -ForegroundColor Green
exit 0
