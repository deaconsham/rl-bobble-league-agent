# T4 live smoke test (Phase 1): six (or N) games at once, crash-safe.
# builds, imports, launches N detached headless Godots each on its own --shm file,
# runs two short epochs with training_loop.py --games N, then a 10-game eval.
# reports turns/min and checks: goals > 0, 0 stuck turns, no shm file clashes,
# out-of-bounds no worse than 1 in 10 games, no python errors.
#
# each Godot must NOT be a child of this shell (it hangs the agent shell), so we
# launch them detached through WMI (Win32_Process.Create) and poll the log files.
# we track and stop only the pids we started.
#
# usage:  powershell -ExecutionPolicy Bypass -File scripts\smoke_test_phase1.ps1
#         -Games 6   (drop to 4 if free RAM is tight)

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 6,
    [int]$Epochs = 2,
    [int]$StepsPerEpoch = 1200,   # total across games (200 each with 6)
    [int]$EvalEpisodes = 10,
    [int]$TimeScale = 32
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$csproj = Join-Path $project "BobbleLeague.csproj"
$logDir = Join-Path $model "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Fail($msg) { Write-Host "SMOKE FAIL: $msg" -ForegroundColor Red; exit 1 }

# ---- 1. dotnet build ----
Write-Host "== dotnet build =="
$csprojBefore = Get-Content -LiteralPath $csproj -Raw
& dotnet build $csproj 2>&1 | Tee-Object -FilePath (Join-Path $logDir "p1_build.log")
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

# ---- 3. train over N games; training_loop --launch owns the game lifecycle ----
# it launches N detached headless Godots (each on its own --shm file + log via
# Game_launcher), trains, restarts any that crash/hang, and stops them at the end.
Write-Host "== train $Epochs epochs x $StepsPerEpoch total steps over $Games games (training owns launch) =="
$trainLog = Join-Path $logDir "p1_train.log"
$evalLog = Join-Path $logDir "p1_eval.log"

# clear old per-game shm + godot logs so the checks only see this run
for ($i = 0; $i -lt $Games; $i++) {
    $shm = Join-Path $env:TEMP "bobble_league_mmap_$i.dat"
    if (Test-Path $shm) { Remove-Item $shm -Force }
    $gl = Join-Path $logDir "godot_$i.log"
    if (Test-Path $gl) { Remove-Item $gl -Force }
}

Push-Location $model
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& python training_loop.py --games $Games --launch --timescale $TimeScale `
    --epochs $Epochs --steps-per-epoch $StepsPerEpoch `
    --opponent chaser --shm-prefix $env:TEMP --log-dir runs/p1_smoke 2>&1 |
    Out-File -Encoding utf8 $trainLog
$trainCode = $LASTEXITCODE
$sw.Stop()
Pop-Location
if ($trainCode -ne 0) { Fail "training exited $trainCode (see $trainLog)" }

$minutes = $sw.Elapsed.TotalMinutes
$totalTurns = $Epochs * $StepsPerEpoch
$turnsPerMin = if ($minutes -gt 0) { [math]::Round($totalTurns / $minutes) } else { 0 }

# per-game shm files must be distinct (no clashes)
$shmPaths = @()
for ($i = 0; $i -lt $Games; $i++) { $shmPaths += (Join-Path $env:TEMP "bobble_league_mmap_$i.dat") }
$distinct = ($shmPaths | Sort-Object -Unique).Count
if ($distinct -ne $Games) { Fail "shm file clash: $distinct distinct of $Games" }

# ---- 4. eval vs chaser: launch one dedicated game, eval, stop only it ----
Write-Host "== eval $EvalEpisodes games vs chaser (one dedicated game) =="
$evalShm = Join-Path $env:TEMP "bobble_league_smoke_eval_mmap.dat"
if (Test-Path $evalShm) { Remove-Item $evalShm -Force }
$evalGodotLog = Join-Path $logDir "p1_eval_godot.log"
$evalArgs = "`"$GodotExe`" --path `"$project`" --headless -- --mode=ai_vs_ai --timescale=$TimeScale --shm=`"$evalShm`""
$evalBat = Join-Path $logDir "run_eval_godot.bat"
Set-Content -LiteralPath $evalBat -Value "@echo off`r`n$evalArgs > `"$evalGodotLog`" 2>&1`r`n" -NoNewline
$ep = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"`"$evalBat`"`"" }
$evalPid = [int]$ep.ProcessId
$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path $evalShm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }

function Stop-EvalGodot {
    # WMI Create returns the cmd wrapper pid, not Godot's, so kill by the unique shm path
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like "*--shm=$evalShm*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

if (-not (Test-Path $evalShm)) { Stop-EvalGodot; Fail "eval game never handshook" }

try {
    Push-Location $model
    $env:BOBBLE_SHM_PATH = $evalShm
    & python evaluate.py --opponent chaser --episodes $EvalEpisodes 2>&1 | Out-File -Encoding utf8 $evalLog
    $evalCode = $LASTEXITCODE
    Remove-Item Env:BOBBLE_SHM_PATH -ErrorAction SilentlyContinue
    Pop-Location
    if ($evalCode -ne 0) { Fail "evaluate exited $evalCode" }
}
finally {
    Stop-EvalGodot
    Write-Host "  stopped eval godot (shm $evalShm)"
}

# ---- 5. gather the numbers ----
$trainText = Get-Content $trainLog -Raw
$godotText = ""
for ($i = 0; $i -lt $Games; $i++) {
    $gLog = Join-Path $logDir "godot_$i.log"
    if (Test-Path $gLog) { $godotText += (Get-Content $gLog -Raw) + "`n" }
}
if (Test-Path $evalGodotLog) { $godotText += (Get-Content $evalGodotLog -Raw) + "`n" }
$allText = $trainText + "`n" + $godotText

$wins = ([regex]::Matches($trainText, "W (\d+)")).Count
# count goals from the per-epoch "W x / L y" lines
$goals = 0
foreach ($m in [regex]::Matches($trainText, "W (\d+) / L (\d+)")) {
    $goals += [int]$m.Groups[1].Value + [int]$m.Groups[2].Value
}
$stuck = ([regex]::Matches($godotText, "did not settle")).Count
$restarts = ([regex]::Matches($trainText, "Vector_env: game \d+ ")).Count
$oob = ([regex]::Matches($godotText, "left the pitch")).Count
$games_played = [math]::Max(1, $goals)
$oobRate = $oob / $games_played
$pyErr = ([regex]::Matches($allText, "Traceback \(most recent call last\)")).Count

Write-Host ""
Write-Host "== phase 1 smoke results ==" -ForegroundColor Cyan
Write-Host "  games:              $Games"
Write-Host "  total turns:        $totalTurns in $([math]::Round($minutes,2)) min"
Write-Host "  turns/min:          $turnsPerMin  (single-game baseline ~470)"
Write-Host "  goals (W+L):        $goals"
Write-Host "  stuck turns:        $stuck"
Write-Host "  restarts:           $restarts"
Write-Host ("  out of bounds:      {0}  (rate {1:P0} of {2} games)" -f $oob, $oobRate, $games_played)
Write-Host "  python tracebacks:  $pyErr"

if ($pyErr -gt 0) { Fail "python error in the run" }
if ($goals -le 0) { Fail "no goals scored" }
if ($stuck -ne 0) { Fail "$stuck stuck turn(s)" }
if ($oobRate -gt 0.15) { Fail ("out-of-bounds rate {0:P0} over the 1-in-10 bar" -f $oobRate) }

Write-Host ""
Write-Host ("SMOKE PASS: games={0} turns/min={1} goals={2} stuck={3} restarts={4} oob={5}" -f `
    $Games, $turnsPerMin, $goals, $stuck, $restarts, $oob) -ForegroundColor Green
exit 0
