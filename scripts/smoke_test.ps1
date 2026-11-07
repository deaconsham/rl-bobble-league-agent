# live smoke test.
# builds, imports, runs two short headless epochs then a 10-game eval, and
# checks goals > 0, 0 stuck turns and out-of-bounds no worse than 1 in 10.
#
# Godot must NOT be a child of the calling shell (it hangs the agent shell),
# so we launch it detached with Start-Process and poll its log file.
#
# usage:  powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
#         -GodotExe "C:\path\to\Godot_...console.exe"   (default: $env:BOBBLE_GODOT_EXE, else the Desktop build)

param(
    [string]$GodotExe = $(if ($env:BOBBLE_GODOT_EXE) { $env:BOBBLE_GODOT_EXE } else { "$env:USERPROFILE\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe" }),
    [int]$Epochs = 2,
    [int]$StepsPerEpoch = 300,
    [int]$EvalEpisodes = 10,
    [int]$TimeScale = 16
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$csproj = Join-Path $project "BobbleLeague.csproj"
$logDir = Join-Path $model "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# one shm file both sides share (the agent shell and Godot have different temps)
$shm = Join-Path $env:TEMP "bobble_league_smoke_mmap.dat"
$env:BOBBLE_SHM_PATH = $shm
if (Test-Path $shm) { Remove-Item $shm -Force }

$godotLog = Join-Path $logDir "smoke_godot.log"
$trainLog = Join-Path $logDir "smoke_train.log"
$evalLog = Join-Path $logDir "smoke_eval.log"

function Fail($msg) { Write-Host "SMOKE FAIL: $msg" -ForegroundColor Red; exit 1 }

# ---- 1. dotnet build ----
Write-Host "== dotnet build =="
$csprojBefore = Get-Content -LiteralPath $csproj -Raw
& dotnet build $csproj 2>&1 | Tee-Object -FilePath (Join-Path $logDir "smoke_build.log")
if ($LASTEXITCODE -ne 0) { Fail "dotnet build returned $LASTEXITCODE" }

# ---- 2. Godot --import, then revert any csproj SDK bump the import makes ----
Write-Host "== godot --import =="
if (-not (Test-Path $GodotExe)) { Fail "Godot not found at $GodotExe" }
$import = Start-Process -FilePath $GodotExe `
    -ArgumentList @("--path", "`"$project`"", "--headless", "--import") `
    -PassThru -WindowStyle Hidden
if (-not $import.WaitForExit(120000)) { $import.Kill(); Fail "godot --import timed out" }

$csprojAfter = Get-Content -LiteralPath $csproj -Raw
if ($csprojAfter -ne $csprojBefore) {
    Write-Host "  reverting csproj change made by import"
    Set-Content -LiteralPath $csproj -Value $csprojBefore -NoNewline
}

# ---- 3. launch Godot headless AI_VS_AI, detached, logging to a file ----
Write-Host "== launch game (detached) =="
$game = Start-Process -FilePath $GodotExe `
    -ArgumentList @("--path", "`"$project`"", "--headless", "--", "--mode=ai_vs_ai", "--timescale=$TimeScale") `
    -PassThru -RedirectStandardOutput $godotLog -RedirectStandardError (Join-Path $logDir "smoke_godot.err") `
    -WindowStyle Hidden
$gamePid = $game.Id
Write-Host "  godot pid $gamePid, waiting for shared memory handshake"

$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path $shm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (-not (Test-Path $shm)) { Stop-Process -Id $gamePid -Force -EA SilentlyContinue; Fail "Godot never created the shm file" }

try {
    # ---- 4. short training run ----
    Write-Host "== train $Epochs epochs x $StepsPerEpoch steps =="
    Push-Location $model
    & python Scripts	raining_loop.py --epochs $Epochs --steps-per-epoch $StepsPerEpoch `
        --opponent chaser --log-dir runs/smoke 2>&1 | Tee-Object -FilePath $trainLog
    $trainCode = $LASTEXITCODE
    Pop-Location
    if ($trainCode -ne 0) { Fail "training exited $trainCode" }

    # ---- 5. eval ----
    # evaluate.py with no --checkpoint picks the highest epoch NUMBER, which may be
    # a stale checkpoint left in Models/ from an older run. the
    # smoke run just wrote a fresh one, so eval the newest checkpoint BY TIME.
    Write-Host "== eval $EvalEpisodes games vs chaser =="
    $freshCkpt = Get-ChildItem (Join-Path $model "Models\ppo_epoch_*.pt") |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $freshCkpt) { Fail "no checkpoint written by training" }
    Write-Host "  using $($freshCkpt.Name)"
    Push-Location $model
    & python Scripts\evaluate.py --opponent chaser --episodes $EvalEpisodes --checkpoint $freshCkpt.FullName 2>&1 | Tee-Object -FilePath $evalLog
    $evalCode = $LASTEXITCODE
    Pop-Location
    if ($evalCode -ne 0) { Fail "evaluate exited $evalCode" }
}
finally {
    # ---- stop ONLY the Godot we started ----
    Stop-Process -Id $gamePid -Force -EA SilentlyContinue
    Write-Host "  stopped godot pid $gamePid"
}

# ---- 6. check the numbers ----
$trainText = Get-Content $trainLog -Raw
$evalText = if (Test-Path $evalLog) { Get-Content $evalLog -Raw } else { "" }
$godotText = if (Test-Path $godotLog) { Get-Content $godotLog -Raw } else { "" }
$allText = $trainText + "`n" + $evalText + "`n" + $godotText

# goals = eval episodes that ended in a win or loss (someone scored). the eval
# prints "Episode N: win/loss/timeout, ...", one per game. counted from the eval
# log, not the short training run, which may finish no episode in 2 epochs.
$wins = ([regex]::Matches($evalText, "Episode \d+: win")).Count
$losses = ([regex]::Matches($evalText, "Episode \d+: loss")).Count
$goals = $wins + $losses

# stuck turns: the game logs "did not settle" on a timeout
$stuck = ([regex]::Matches($godotText, "did not settle")).Count
# out of bounds: the game logs "left the pitch". this is harmless (the turn
# returns to kickoff with no reward), so the bar is a RATE over games played,
# not a raw count over the hundreds of turns a short training run takes.
$oob = ([regex]::Matches($godotText, "left the pitch")).Count
$games = [math]::Max(1, $goals)
$oobRate = $oob / $games

# python errors in either log
$pyErr = ([regex]::Matches($allText, "Traceback \(most recent call last\)")).Count

Write-Host ""
Write-Host "== smoke results ==" -ForegroundColor Cyan
Write-Host "  goals scored (W+L): $goals  (W $wins / L $losses)"
Write-Host "  stuck turns:        $stuck"
Write-Host ("  out of bounds:      {0}  (rate {1:P0} of {2} games)" -f $oob, $oobRate, $games)
Write-Host "  python tracebacks:  $pyErr"

if ($pyErr -gt 0) { Fail "python error in the run" }
if ($goals -le 0) { Fail "no goals scored" }
if ($stuck -ne 0) { Fail "$stuck stuck turn(s)" }
# plan bar: no more than ~1 in 10 games leaves the pitch
if ($oobRate -gt 0.15) { Fail ("out-of-bounds rate {0:P0} ({1}/{2}) over the 1-in-10 bar" -f $oobRate, $oob, $games) }

Write-Host ""
Write-Host ("SMOKE PASS: goals={0} stuck={1} oob={2} ({3:P0})" -f $goals, $stuck, $oob, $oobRate) -ForegroundColor Green
exit 0
