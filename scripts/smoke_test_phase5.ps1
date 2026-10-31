# T4 live smoke test (Phase 5): live-vs-live games, both teams' data, turn length.
# Same shape as the Phase 1/2 smokes (build, import, N detached games, short
# train, eval, number checks) but trains with --opponent self so the live slot
# plays, Team B turns are stored, and the turn-length histogram is produced.
# The Python bridge refuses anything but protocol v4, so a successful handshake
# already proves Godot is speaking v4.
#
# each Godot must NOT be a child of this shell (it hangs the agent shell), so we
# launch detached through WMI (Win32_Process.Create) and poll the log files.
# we track and stop only the pids we started.
#
# usage:  powershell -ExecutionPolicy Bypass -File scripts\smoke_test_phase5.ps1
#         -Games 4   (6 if free RAM allows)

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 4,
    [int]$Epochs = 2,
    [int]$StepsPerEpoch = 1200,   # total across games
    [int]$EvalEpisodes = 10,
    [int]$TimeScale = 32,
    [string]$LogTag = "p5_smoke"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$csproj = Join-Path $project "BobbleLeague.csproj"
$logDir = Join-Path $model "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Fail($msg) { Write-Host "SMOKE FAIL: $msg" -ForegroundColor Red; exit 1 }

# stop any smoke godots a prior aborted run left behind (match our smoke shm names)
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like '*bobble_league_p5smoke_*mmap*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 500

# ---- 1. dotnet build ----
Write-Host "== dotnet build =="
$csprojBefore = Get-Content -LiteralPath $csproj -Raw
& dotnet build $csproj 2>&1 | Tee-Object -FilePath (Join-Path $logDir "p5_build.log")
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

# ---- 3. train over N self-play games; training_loop --launch owns the lifecycle ----
Write-Host "== train $Epochs epochs x $StepsPerEpoch total steps over $Games self-play games =="
$trainLog = Join-Path $logDir "$($LogTag)_train.log"
$evalLog = Join-Path $logDir "$($LogTag)_eval.log"

for ($i = 0; $i -lt $Games; $i++) {
    $shm = Join-Path $env:TEMP "bobble_league_mmap_$i.dat"
    if (Test-Path $shm) { Remove-Item $shm -Force }
    $gl = Join-Path $logDir "godot_$i.log"
    if (Test-Path $gl) { Remove-Item $gl -Force }
}

Push-Location $model
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& python training_loop.py --games $Games --launch --timescale $TimeScale `
    --epochs $Epochs --steps-per-epoch $StepsPerEpoch --snapshot-freq 1 `
    --opponent self --shm-prefix $env:TEMP --log-dir runs/$LogTag 2>&1 |
    Out-File -Encoding utf8 $trainLog
$trainCode = $LASTEXITCODE
$sw.Stop()
Pop-Location
if ($trainCode -ne 0) { Fail "training exited $trainCode (see $trainLog)" }

$minutes = $sw.Elapsed.TotalMinutes
$totalTurns = $Epochs * $StepsPerEpoch
$turnsPerMin = if ($minutes -gt 0) { [math]::Round($totalTurns / $minutes) } else { 0 }

$shmPaths = @()
for ($i = 0; $i -lt $Games; $i++) { $shmPaths += (Join-Path $env:TEMP "bobble_league_mmap_$i.dat") }
$distinct = ($shmPaths | Sort-Object -Unique).Count
if ($distinct -ne $Games) { Fail "shm file clash: $distinct distinct of $Games" }

# ---- helper: run one dedicated headless godot on its own shm, do $Body, stop only it ----
function With-DedicatedGodot($shm, $godotLog, [scriptblock]$Body) {
    if (Test-Path $shm) { Remove-Item $shm -Force }
    $args = "`"$GodotExe`" --path `"$project`" --headless -- --mode=ai_vs_ai --timescale=$TimeScale --shm=`"$shm`""
    $bat = Join-Path $logDir ("run_" + [System.IO.Path]::GetFileNameWithoutExtension($godotLog) + ".bat")
    Set-Content -LiteralPath $bat -Value "@echo off`r`n$args > `"$godotLog`" 2>&1`r`n" -NoNewline
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"`"$bat`"`"" } | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Path $shm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    # match on the shm FILE NAME: WMI stores the command line with the path quoted,
    # so a literal "--shm=$shm" would miss it (the Phase 4 stop bug). The file name
    # is unique per dedicated game, so matching it stops this game and no other.
    $shmLeaf = Split-Path $shm -Leaf
    $stop = {
        Get-CimInstance Win32_Process |
            Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like "*$shmLeaf*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path $shm)) { & $stop; Fail "dedicated godot ($shm) never handshook" }
    try { & $Body } finally { & $stop; Write-Host "  stopped dedicated godot (shm $shm)" }
}

# ---- 4. eval vs chaser on one dedicated game (explicit checkpoint, not a glob) ----
$freshCkpt = "checkpoints/ppo_epoch_$($Epochs - 1).pt"
Write-Host "== eval $EvalEpisodes games vs chaser ($freshCkpt, one dedicated game) =="
$evalShm = Join-Path $env:TEMP "bobble_league_p5smoke_eval_mmap.dat"
$evalGodotLog = Join-Path $logDir "$($LogTag)_eval_godot.log"
With-DedicatedGodot $evalShm $evalGodotLog {
    Push-Location $model
    $env:BOBBLE_SHM_PATH = $evalShm
    & python evaluate.py --checkpoint $freshCkpt --opponent chaser --episodes $EvalEpisodes 2>&1 | Out-File -Encoding utf8 $evalLog
    $script:evalCode = $LASTEXITCODE
    Remove-Item Env:BOBBLE_SHM_PATH -ErrorAction SilentlyContinue
    Pop-Location
    if ($script:evalCode -ne 0) { Fail "evaluate exited $script:evalCode" }
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

# Team B turns stored (live vs live) and turn-length stats, from the train log
$teamB = 0
foreach ($m in [regex]::Matches($trainText, "Team B turns stored \(live-vs-live\): (\d+)")) {
    $teamB += [int]$m.Groups[1].Value
}
$turnLines = @()
foreach ($line in ($trainText -split "`r?`n")) {
    if ($line -match "Turn length \(s\):") { $turnLines += $line.Trim() }
}

Write-Host ""
Write-Host "== phase 5 smoke results ==" -ForegroundColor Cyan
Write-Host "  games:              $Games"
Write-Host "  total turns:        $totalTurns in $([math]::Round($minutes,2)) min"
Write-Host "  turns/min:          $turnsPerMin"
Write-Host "  goals (W+L):        $goals"
Write-Host "  stuck turns:        $stuck"
Write-Host "  restarts:           $restarts"
Write-Host ("  out of bounds:      {0}  (rate {1:P0} of {2} games)" -f $oob, $oobRate, $games_played)
Write-Host "  python tracebacks:  $pyErr"
Write-Host "  Team B turns stored: $teamB"
foreach ($l in $turnLines) { Write-Host "  $l" }

if ($pyErr -gt 0) { Fail "python error in the run" }
if ($goals -le 0) { Fail "no goals scored" }
if ($stuck -ne 0) { Fail "$stuck stuck turn(s)" }
if ($teamB -le 0) { Fail "no Team B turns stored - live-vs-live data path did not run" }
if ($turnLines.Count -le 0) { Fail "no turn-length stats logged" }
if ($oobRate -gt 0.15) { Fail ("out-of-bounds rate {0:P0} over the 1-in-10 bar" -f $oobRate) }

Write-Host ""
Write-Host ("SMOKE PASS: games={0} turns/min={1} goals={2} stuck={3} restarts={4} oob={5} teamB={6}" -f `
    $Games, $turnsPerMin, $goals, $stuck, $restarts, $oob, $teamB) -ForegroundColor Green
exit 0
