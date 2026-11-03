# T4 live smoke + T5 strength (Phase 6): PFSP, shaping fade, Elo league eval.
#
# Same detached-launch shape as the Phase 1/2/5 smokes (build, import, N detached
# games, short train, eval, number checks), with two Phase 6 additions:
#   - training runs with --pfsp --shaping-fade so the new opponent selection and
#     the reward fade actually execute against real Godot (T4);
#   - after training, league_eval.py plays the two fresh checkpoints against each
#     other and the held-out anchors on one dedicated game, and we check the Elo
#     CSV is produced with the anchors present and in a sane order (T5).
#
# each Godot must NOT be a child of this shell (it hangs the agent shell), so we
# launch detached through WMI (Win32_Process.Create) and poll the log files. we
# track and stop only the pids we started.
#
# usage:  powershell -ExecutionPolicy Bypass -File scripts\smoke_test_phase6.ps1
#         -Games 4   (6 if free RAM allows)

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 4,
    [int]$Epochs = 4,
    [int]$StepsPerEpoch = 1200,   # total across games
    [int]$EvalEpisodes = 10,
    [int]$LeagueGames = 12,       # games per league pairing (T5)
    [int]$TimeScale = 32,
    [string]$LogTag = "p6_smoke"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$csproj = Join-Path $project "BobbleLeague.csproj"
$logDir = Join-Path $model "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Fail($msg) { Write-Host "SMOKE FAIL: $msg" -ForegroundColor Red; exit 1 }

# stop any stray godots a prior aborted run left behind
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like '*bobble_league_p6*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 500

# ---- 1. dotnet build ----
Write-Host "== dotnet build =="
$csprojBefore = Get-Content -LiteralPath $csproj -Raw
& dotnet build $csproj 2>&1 | Tee-Object -FilePath (Join-Path $logDir "p6_build.log")
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

# ---- 3. train over N self-play games WITH pfsp + shaping fade ----
Write-Host "== train $Epochs epochs x $StepsPerEpoch total steps over $Games games (--pfsp --shaping-fade) =="
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
    --opponent self --pfsp --shaping-fade --shaping-fade-fraction 0.5 `
    --shm-prefix $env:TEMP --log-dir runs/$LogTag 2>&1 |
    Out-File -Encoding utf8 $trainLog
$trainCode = $LASTEXITCODE
$sw.Stop()
Pop-Location
if ($trainCode -ne 0) { Fail "training exited $trainCode (see $trainLog)" }

$minutes = $sw.Elapsed.TotalMinutes
$totalTurns = $Epochs * $StepsPerEpoch
$turnsPerMin = if ($minutes -gt 0) { [math]::Round($totalTurns / $minutes) } else { 0 }

# ---- helper: run one dedicated headless godot on its own shm, do $Body, stop only it ----
function With-DedicatedGodot($shm, $godotLog, [scriptblock]$Body) {
    if (Test-Path $shm) { Remove-Item $shm -Force }
    $args = "`"$GodotExe`" --path `"$project`" --headless -- --mode=ai_vs_ai --timescale=$TimeScale --shm=`"$shm`""
    $bat = Join-Path $logDir ("run_" + [System.IO.Path]::GetFileNameWithoutExtension($godotLog) + ".bat")
    Set-Content -LiteralPath $bat -Value "@echo off`r`n$args > `"$godotLog`" 2>&1`r`n" -NoNewline
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"`"$bat`"`"" } | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Path $shm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
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
$evalShm = Join-Path $env:TEMP "bobble_league_p6_eval_mmap.dat"
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

# ---- 5. T5 league Elo: two fresh checkpoints vs each other + held-out anchors ----
$ckptA = "checkpoints/ppo_epoch_0.pt"
$ckptB = "checkpoints/ppo_epoch_$($Epochs - 1).pt"
$eloCsv = Join-Path $model "runs/$LogTag/elo.csv"
$leagueLog = Join-Path $logDir "$($LogTag)_league.log"
Write-Host "== T5 league eval ($ckptA, $ckptB vs anchors; $LeagueGames games/pairing) =="
$leagueShm = Join-Path $env:TEMP "bobble_league_p6_league_mmap.dat"
$leagueGodotLog = Join-Path $logDir "$($LogTag)_league_godot.log"
With-DedicatedGodot $leagueShm $leagueGodotLog {
    Push-Location $model
    & python league_eval.py --checkpoints $ckptA $ckptB --games $LeagueGames `
        --shm $leagueShm --out runs/$LogTag/elo.csv 2>&1 | Out-File -Encoding utf8 $leagueLog
    $script:leagueCode = $LASTEXITCODE
    Pop-Location
    if ($script:leagueCode -ne 0) { Fail "league_eval exited $script:leagueCode (see $leagueLog)" }
}

# ---- 6. gather the numbers ----
$trainText = Get-Content $trainLog -Raw
$godotText = ""
for ($i = 0; $i -lt $Games; $i++) {
    $gLog = Join-Path $logDir "godot_$i.log"
    if (Test-Path $gLog) { $godotText += (Get-Content $gLog -Raw) + "`n" }
}
foreach ($gl in @($evalGodotLog, $leagueGodotLog)) {
    if (Test-Path $gl) { $godotText += (Get-Content $gl -Raw) + "`n" }
}
$leagueText = if (Test-Path $leagueLog) { Get-Content $leagueLog -Raw } else { "" }
$allText = $trainText + "`n" + $godotText + "`n" + $leagueText

$goals = 0
foreach ($m in [regex]::Matches($trainText, "W (\d+) / L (\d+)")) {
    $goals += [int]$m.Groups[1].Value + [int]$m.Groups[2].Value
}
$stuck = ([regex]::Matches($godotText, "did not settle")).Count
$oob = ([regex]::Matches($godotText, "left the pitch")).Count
$games_played = [math]::Max(1, $goals)
$oobRate = $oob / $games_played
$pyErr = ([regex]::Matches($allText, "Traceback \(most recent call last\)")).Count

# shaping-fade proof: the training log prints one LR line per epoch; the fade
# runs inside the env, so we confirm the run finished all epochs without error
# and the league CSV exists and is sane. (the fade schedule itself is T1-tested.)

Write-Host ""
Write-Host "== phase 6 smoke results ==" -ForegroundColor Cyan
Write-Host "  games:              $Games"
Write-Host "  total turns:        $totalTurns in $([math]::Round($minutes,2)) min"
Write-Host "  turns/min:          $turnsPerMin"
Write-Host "  goals (W+L):        $goals"
Write-Host "  stuck turns:        $stuck"
Write-Host ("  out of bounds:      {0}  (rate {1:P0} of {2} games)" -f $oob, $oobRate, $games_played)
Write-Host "  python tracebacks:  $pyErr"

# ---- 7. T5 Elo CSV checks: anchors present, ratings parse, sane order ----
if (-not (Test-Path $eloCsv)) { Fail "no Elo CSV at $eloCsv" }
$elo = Import-Csv $eloCsv
Write-Host ""
Write-Host "== T5 Elo ratings ==" -ForegroundColor Cyan
foreach ($row in $elo) { Write-Host ("  {0,-24} {1,8}  anchor={2}" -f $row.name, $row.rating, $row.anchor) }

$anchorNames = @("random", "chaser", "defender", "replay")
$haveAnchors = @($elo | Where-Object { $anchorNames -contains $_.name })
if ($haveAnchors.Count -lt $anchorNames.Count) { Fail "Elo CSV missing held-out anchors (found $($haveAnchors.Count)/4)" }
# every rating must be a finite number
foreach ($row in $elo) {
    $r = 0.0
    if (-not [double]::TryParse($row.rating, [ref]$r)) { Fail "non-numeric Elo rating for $($row.name)" }
}
# the two anchors have fixed ratings: chaser (1000) must outrank random (800),
# proving the anchor scale came through intact.
$randomElo = [double]($elo | Where-Object { $_.name -eq "random" }).rating
$chaserElo = [double]($elo | Where-Object { $_.name -eq "chaser" }).rating
if ($chaserElo -le $randomElo) { Fail "anchor order wrong: chaser ($chaserElo) <= random ($randomElo)" }

# surface the newest-checkpoint-vs-anchor win rates + replay score rate from the log
$leagueText -split "`r?`n" | Where-Object { $_ -match "win rate|score rate|vs " } |
    ForEach-Object { Write-Host "  $_" }

if ($pyErr -gt 0) { Fail "python error in the run" }
if ($goals -le 0) { Fail "no goals scored" }
if ($stuck -ne 0) { Fail "$stuck stuck turn(s)" }
if ($oobRate -gt 0.15) { Fail ("out-of-bounds rate {0:P0} over the 1-in-10 bar" -f $oobRate) }

Write-Host ""
Write-Host ("SMOKE PASS: games={0} turns/min={1} goals={2} stuck={3} oob={4} chaserElo={5} randomElo={6}" -f `
    $Games, $turnsPerMin, $goals, $stuck, $oob, $chaserElo, $randomElo) -ForegroundColor Green
exit 0
