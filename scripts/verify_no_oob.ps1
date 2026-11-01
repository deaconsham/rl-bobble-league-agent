# live ball-leak check: play N games headless and count how many times the ball
# leaves the pitch. the cage + continuous collision added in phase 8 should make
# this 0. launches its own dedicated godot on a private shm so it never touches a
# training or eval run that is already going.
#
# godot must not be a child of the calling shell, so it is launched detached via
# Invoke-CimMethod Win32_Process Create through a per-run .bat (absolute paths),
# and we poll its log. stop only the godot we started, matched by its shm token.
#
# usage:  powershell -ExecutionPolicy Bypass -File scripts\verify_no_oob.ps1 -Games 200

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 200,
    [int]$TimeScale = 32,
    [string]$Opponent = "chaser",
    [string]$Checkpoint = "p7_run/checkpoints/ppo_epoch_60.pt"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$logDir = Join-Path $model "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$shm = Join-Path $env:TEMP "bobble_league_oobcheck.dat"
$env:BOBBLE_SHM_PATH = $shm
if (Test-Path $shm) { Remove-Item $shm -Force }

$godotLog = Join-Path $logDir "oob_godot.log"
$playLog = Join-Path $logDir "oob_play.log"
$bat = Join-Path $logDir "oob_godot.bat"
if (Test-Path $godotLog) { Remove-Item $godotLog -Force }

function Fail($msg) { Write-Host "OOB CHECK FAIL: $msg" -ForegroundColor Red; exit 1 }

# ---- launch godot detached through a .bat (absolute paths, redirection) ----
$batBody = "`"$GodotExe`" --path `"$project`" --headless -- --mode=ai_vs_ai --timescale=$TimeScale --shm=$shm > `"$godotLog`" 2>&1"
Set-Content -LiteralPath $bat -Value $batBody -Encoding ascii
Write-Host "== launch godot (detached) on $shm =="
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"$bat`"" } | Out-Null

$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path $shm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (-not (Test-Path $shm)) { Fail "godot never created the shm file" }
Start-Sleep -Seconds 3

try {
    Write-Host "== play $Games games vs $Opponent =="
    Push-Location $model
    & python evaluate.py --opponent $Opponent --episodes $Games --checkpoint $Checkpoint 2>&1 | Tee-Object -FilePath $playLog
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Fail "evaluate exited $code" }
}
finally {
    # stop only the godot we started, matched by the unique shm token
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "oobcheck" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
}

$godotText = if (Test-Path $godotLog) { Get-Content $godotLog -Raw } else { "" }
$oob = ([regex]::Matches($godotText, "left the pitch")).Count
$stuck = ([regex]::Matches($godotText, "did not settle")).Count

Write-Host ""
Write-Host "== oob results ==" -ForegroundColor Cyan
Write-Host "  games played:  $Games"
Write-Host "  balls out:     $oob"
Write-Host "  stuck turns:   $stuck"
if ($oob -gt 0) {
    Write-Host "  exit spots:"
    [regex]::Matches($godotText, "ball out at .*") | Select-Object -First 20 | ForEach-Object { "    " + $_.Value }
}

if ($oob -ne 0) { Fail "$oob ball(s) left the pitch over $Games games" }
Write-Host ""
Write-Host ("OOB CHECK PASS: 0 balls out over {0} games (stuck {1})" -f $Games, $stuck) -ForegroundColor Green
exit 0
