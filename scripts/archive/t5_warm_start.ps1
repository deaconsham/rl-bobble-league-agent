# T5 strength check (Phase 4): the warm-started net, BEFORE any PPO, beats a
# random-init net when both play the chaser. We:
#   1. record chaser demos over N real games and behaviour-clone -> warm_start.pt,
#   2. save a fresh random-init net in the same format -> random_init.pt (no Godot),
#   3. evaluate each vs the chaser over >= Episodes games on one dedicated Godot,
#   4. report win rate and goal diff (mean return) for both and PASS if warm wins
#      more and has a higher goal diff than random.
#
# Godot must NOT be a child of this shell (it hangs the agent shell), so each
# game is launched detached through WMI and we stop only the pids we started.
#
# usage:  powershell -ExecutionPolicy Bypass -File scripts\t5_warm_start.ps1
#         -Games 4 -RecordTurns 6000 -Episodes 20   (4 games: RAM is tight)

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 4,
    [int]$RecordTurns = 6000,
    [int]$CloneEpochs = 10,
    [int]$Episodes = 20,
    [int]$TimeScale = 32
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$logDir = Join-Path $model "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$warmCkpt = "checkpoints/warm_start.pt"
$randCkpt = "checkpoints/random_init.pt"

function Fail($msg) { Write-Host "T5 FAIL: $msg" -ForegroundColor Red; exit 1 }

# stop any T5 godots a prior aborted run left behind
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like '*bobble_league_t5_*mmap*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 500

# ---- 1. record chaser demos over N real games and behaviour-clone ----
Write-Host "== warm start: record $RecordTurns chaser turns over $Games games + clone =="
for ($i = 0; $i -lt $Games; $i++) {
    $shm = Join-Path $env:TEMP "bobble_league_mmap_$i.dat"
    if (Test-Path $shm) { Remove-Item $shm -Force }
    $gl = Join-Path $logDir "godot_$i.log"
    if (Test-Path $gl) { Remove-Item $gl -Force }
}
$warmLog = Join-Path $logDir "p4_warm.log"
Push-Location $model
& python warm_start.py --games $Games --turns $RecordTurns --clone-epochs $CloneEpochs `
    --timescale $TimeScale --out $warmCkpt --shm-prefix $env:TEMP 2>&1 |
    Out-File -Encoding utf8 $warmLog
$warmCode = $LASTEXITCODE
Pop-Location
if ($warmCode -ne 0) { Fail "warm_start exited $warmCode (see $warmLog)" }
if (-not (Test-Path (Join-Path $model $warmCkpt))) { Fail "no warm checkpoint written" }

# ---- 2. save the random-init baseline (no Godot) ----
Write-Host "== save random-init baseline =="
Push-Location $model
& python warm_start.py --random-init $randCkpt 2>&1 | Out-File -Encoding utf8 -Append $warmLog
$randCode = $LASTEXITCODE
Pop-Location
if ($randCode -ne 0) { Fail "random-init save exited $randCode" }

# ---- helper: one dedicated headless godot on its own shm, run $Body, stop only it ----
function With-DedicatedGodot($shm, $godotLog, [scriptblock]$Body) {
    if (Test-Path $shm) { Remove-Item $shm -Force }
    $gargs = "`"$GodotExe`" --path `"$project`" --headless -- --mode=ai_vs_ai --timescale=$TimeScale --shm=`"$shm`""
    $bat = Join-Path $logDir ("run_" + [System.IO.Path]::GetFileNameWithoutExtension($godotLog) + ".bat")
    Set-Content -LiteralPath $bat -Value "@echo off`r`n$gargs > `"$godotLog`" 2>&1`r`n" -NoNewline
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"`"$bat`"`"" } | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Path $shm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    $stop = {
        $leaf = [System.IO.Path]::GetFileName($shm)
        Get-CimInstance Win32_Process |
            Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like "*$leaf*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path $shm)) { & $stop; Fail "dedicated godot ($shm) never handshook" }
    try { & $Body } finally { & $stop; Write-Host "  stopped dedicated godot (shm $shm)" }
}

# ---- helper: eval one checkpoint vs chaser over $Episodes games, return W/L/T + mean return ----
function Eval-Checkpoint($ckpt, $tag) {
    $shm = Join-Path $env:TEMP "bobble_league_t5_${tag}_mmap.dat"
    $godotLog = Join-Path $logDir "p4_t5_${tag}_godot.log"
    $evalLog = Join-Path $logDir "p4_t5_${tag}.log"
    With-DedicatedGodot $shm $godotLog {
        Push-Location $model
        $env:BOBBLE_SHM_PATH = $shm
        & python evaluate.py --checkpoint $ckpt --opponent chaser --episodes $Episodes 2>&1 |
            Out-File -Encoding utf8 $evalLog
        $script:evalCode = $LASTEXITCODE
        Remove-Item Env:BOBBLE_SHM_PATH -ErrorAction SilentlyContinue
        Pop-Location
        if ($script:evalCode -ne 0) { Fail "evaluate ($tag) exited $script:evalCode (see $evalLog)" }
    }
    $text = Get-Content $evalLog -Raw
    $m = [regex]::Match($text, "W (\d+) / L (\d+) / T (\d+)\s+win rate (\d+)%, mean return ([\-\d\.]+)")
    if (-not $m.Success) { Fail "could not parse eval output for $tag (see $evalLog)" }
    return [pscustomobject]@{
        Tag        = $tag
        Win        = [int]$m.Groups[1].Value
        Loss       = [int]$m.Groups[2].Value
        Timeout    = [int]$m.Groups[3].Value
        WinRate    = [int]$m.Groups[4].Value
        MeanReturn = [double]$m.Groups[5].Value
    }
}

# ---- 3. evaluate warm vs chaser, then random vs chaser ----
Write-Host "== eval warm-started net vs chaser ($Episodes games) =="
$warm = Eval-Checkpoint $warmCkpt "warm"
Write-Host "== eval random-init net vs chaser ($Episodes games) =="
$rand = Eval-Checkpoint $randCkpt "random"

# goal diff per game ~= mean return (reward is goal-dominated)
Write-Host ""
Write-Host "== phase 4 T5 results ($Episodes games each, before any PPO) ==" -ForegroundColor Cyan
Write-Host ("  warm   : W {0} / L {1} / T {2}   win rate {3}%   goal diff/game {4:N3}" -f `
    $warm.Win, $warm.Loss, $warm.Timeout, $warm.WinRate, $warm.MeanReturn)
Write-Host ("  random : W {0} / L {1} / T {2}   win rate {3}%   goal diff/game {4:N3}" -f `
    $rand.Win, $rand.Loss, $rand.Timeout, $rand.WinRate, $rand.MeanReturn)

if ($warm.WinRate -le $rand.WinRate) {
    Fail ("warm win rate {0}% not above random {1}%" -f $warm.WinRate, $rand.WinRate)
}
if ($warm.MeanReturn -le $rand.MeanReturn) {
    Fail ("warm goal diff {0:N3} not above random {1:N3}" -f $warm.MeanReturn, $rand.MeanReturn)
}

Write-Host ""
Write-Host ("T5 PASS: warm beats random vs chaser (win {0}% vs {1}%, goal diff {2:N3} vs {3:N3})" -f `
    $warm.WinRate, $rand.WinRate, $warm.MeanReturn, $rand.MeanReturn) -ForegroundColor Green
exit 0
