# T4 live smoke on 6 games at one time scale (Phase 9).
#
# Lets training_loop.py launch its own 6 detached headless Godot games (--launch,
# the real training path: WMI-detached, never children of this shell, stopped by
# the Vector_env on close). Runs a short 6-game training burst, then an eval, and
# reports turns/min, stuck turns, the "did not settle" rate and out-of-bounds.
#
# usage:
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_test_phase9.ps1 -TimeScale 48
#
# turns/min is the whole-run measure the plan uses: one training step plays one
# turn on every game, so total game-turns = StepsPerEpoch * Epochs over the wall
# clock of the training burst.

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 6,
    [int]$TimeScale = 48,
    [int]$Epochs = 2,
    [int]$StepsPerEpoch = 1200,
    [int]$EvalEpisodes = 10
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$logDir = Join-Path $model "logs\p9_t4_ts$TimeScale"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Fail($msg) { Write-Host "P9 T4 FAIL: $msg" -ForegroundColor Red; exit 1 }

$trainLog = Join-Path $logDir "train.log"
$evalLog = Join-Path $logDir "eval.log"
$shmPrefix = Join-Path $env:TEMP "p9_ts$TimeScale"
New-Item -ItemType Directory -Force -Path $shmPrefix | Out-Null

# ---- short 6-game training burst (training_loop launches + stops the games) ----
Write-Host "== train $Epochs x $StepsPerEpoch over $Games games at timescale x$TimeScale =="
$sw = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $model
& python training_loop.py --games $Games --steps-per-epoch $StepsPerEpoch `
    --epochs $Epochs --opponent chaser --launch --godot $GodotExe `
    --project $project --shm-prefix $shmPrefix --timescale $TimeScale `
    --log-dir runs/p9_t4_ts$TimeScale 2>&1 | Tee-Object -FilePath $trainLog
$trainCode = $LASTEXITCODE
Pop-Location
$sw.Stop()
if ($trainCode -ne 0) { Fail "training exited $trainCode" }
$minutes = $sw.Elapsed.TotalMinutes
$totalTurns = $StepsPerEpoch * $Epochs
$turnsPerMin = if ($minutes -gt 0) { [math]::Round($totalTurns / $minutes) } else { 0 }

# the launcher logs its games under bobble_model/logs/godot_<i>.log
$godotText = ""
for ($i = 0; $i -lt $Games; $i++) {
    $g = Join-Path $model "logs\godot_$i.log"
    if (Test-Path $g) { $godotText += (Get-Content $g -Raw) + "`n" }
}

# ---- eval the freshest checkpoint on its own game ----
Write-Host "== eval $EvalEpisodes games vs chaser =="
$fresh = Get-ChildItem (Join-Path $model "checkpoints\ppo_epoch_*.pt") |
    Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $fresh) { Fail "no checkpoint written by training" }
$evalShm = Join-Path $env:TEMP "bobble_league_p9_eval.dat"
if (Test-Path $evalShm) { Remove-Item $evalShm -Force }
$evalGodotLog = Join-Path $logDir "eval_godot.log"
$evalBat = Join-Path $logDir "run_eval_godot.bat"
$cmd = "`"$GodotExe`" --path `"$project`" --headless -- --mode=ai_vs_ai --timescale=$TimeScale `"--shm=$evalShm`""
Set-Content -LiteralPath $evalBat -Value "@echo off`r`n$cmd > `"$evalGodotLog`" 2>&1`r`n" -NoNewline
# detached enough: the cmd wrapper runs the bat then exits, Godot keeps running.
# a single quote level (cmd /c "<bat>") passes the quoted --shm path through intact,
# which the doubled-quote WMI form mangled when the path has a space.
Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$evalBat`"") -WindowStyle Hidden | Out-Null
$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path $evalShm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (-not (Test-Path $evalShm)) { Fail "eval Godot never created its shm" }
Start-Sleep -Seconds 2
try {
    $env:BOBBLE_SHM_PATH = $evalShm
    Push-Location $model
    & python evaluate.py --opponent chaser --episodes $EvalEpisodes --checkpoint $fresh.FullName 2>&1 | Tee-Object -FilePath $evalLog
    Pop-Location
}
finally {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like "*--shm=$evalShm*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
if (Test-Path $evalGodotLog) { $godotText += (Get-Content $evalGodotLog -Raw) + "`n" }

# ---- parse the numbers ----
$trainText = Get-Content $trainLog -Raw
$evalText = if (Test-Path $evalLog) { Get-Content $evalLog -Raw } else { "" }

$stuck = ([regex]::Matches($godotText, "did not settle")).Count
$oob = ([regex]::Matches($godotText, "left the pitch")).Count
$pyErr = ([regex]::Matches(($trainText + $evalText + $godotText), "Traceback \(most recent call last\)")).Count
# AI_VS_AI does not print a per-turn line, so the did-not-settle rate is measured
# against the turns the training burst actually played (one per step per game).
$settleRate = if ($totalTurns -gt 0) { $stuck / $totalTurns } else { 0 }

$wins = ([regex]::Matches($evalText, "Episode \d+: win")).Count
$losses = ([regex]::Matches($evalText, "Episode \d+: loss")).Count
$goals = $wins + $losses

Write-Host ""
Write-Host "== P9 T4 results (timescale x$TimeScale, $Games games) ==" -ForegroundColor Cyan
Write-Host ("  turns/min:          {0}  ({1} turns in {2:N2} min)" -f $turnsPerMin, $totalTurns, $minutes)
Write-Host ("  stuck / did-not-settle: {0}  (rate {1:P2} of {2} turns)" -f $stuck, $settleRate, $totalTurns)
Write-Host ("  out of bounds:      {0}" -f $oob)
Write-Host ("  eval goals (W+L):   {0}  (W {1} / L {2})" -f $goals, $wins, $losses)
Write-Host ("  python tracebacks:  {0}" -f $pyErr)

if ($pyErr -gt 0) { Fail "python error in the run" }
if ($stuck -ne 0) { Fail "$stuck stuck turn(s)" }
if ($settleRate -gt 0.01) { Fail ("did-not-settle rate {0:P2} over the 1% bar" -f $settleRate) }

Write-Host ""
Write-Host ("P9 T4 PASS (ts x{0}): turns/min={1} stuck={2} settleRate={3:P2} oob={4}" -f $TimeScale, $turnsPerMin, $stuck, $settleRate, $oob) -ForegroundColor Green
exit 0
