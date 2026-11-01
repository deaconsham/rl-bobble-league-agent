# phase 8 step 2: re-test ppo_epoch_60 over 40 games each vs the fixed anchors
# (chaser 1000, defender 950, replay 850, random 800) to place it on the elo
# scale, and 40 games vs warm_start.pt. the resulting elo is the bar later phases
# must beat. launches its own dedicated godot on a private shm (does not touch a
# training or other eval already running) and stops only the godot it started.
#
# usage:  powershell -ExecutionPolicy Bypass -File scripts\eval_epoch60.ps1

param(
    [string]$GodotExe = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe",
    [int]$Games = 40
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo "bobble_league"
$model = Join-Path $repo "bobble_model"
$logDir = Join-Path $model "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$shm = Join-Path $env:TEMP "bobble_league_e60eval.dat"
$env:BOBBLE_SHM_PATH = $shm
if (Test-Path $shm) { Remove-Item $shm -Force }

$godotLog = Join-Path $logDir "e60_godot.log"
$outLog = Join-Path $logDir "e60_eval.log"
$bat = Join-Path $logDir "e60_godot.bat"
if (Test-Path $godotLog) { Remove-Item $godotLog -Force }

$e60 = Join-Path $model "p7_run\checkpoints\ppo_epoch_60.pt"
$warm = Join-Path $model "p7_run\checkpoints\warm_start.pt"
if (-not (Test-Path $warm)) { $warm = Join-Path $model "checkpoints\warm_start.pt" }
$eloCsv = Join-Path $model "p7_run\elo_epoch60.csv"

# ---- launch godot detached through a .bat (absolute paths) ----
$batBody = "`"$GodotExe`" --path `"$project`" --headless -- --mode=ai_vs_ai --timescale=32 --shm=$shm > `"$godotLog`" 2>&1"
Set-Content -LiteralPath $bat -Value $batBody -Encoding ascii
Write-Host "== launch godot (detached) on $shm =="
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"$bat`"" } | Out-Null

$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path $shm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (-not (Test-Path $shm)) { Write-Host "FAIL: godot never created shm"; exit 1 }
Start-Sleep -Seconds 3

try {
    Push-Location $model
    # 1. elo of epoch 60 against the fixed anchors (40 games each) -> the bar
    Write-Host "== epoch 60 elo vs anchors ($Games games each) =="
    & python league_eval.py --checkpoints "p7_run/checkpoints/ppo_epoch_60.pt" `
        --games $Games --shm $shm --out $eloCsv 2>&1 | Tee-Object -FilePath $outLog
    # 2. epoch 60 vs the warm start, 40 games (learner = epoch 60, opponent = warm)
    Write-Host "== epoch 60 vs warm_start ($Games games) =="
    & python -c @"
import os
from bobble_league_env import Bobble_league_env
from opponents import PolicyOpponent, load_state_dict
e60 = r'p7_run/checkpoints/ppo_epoch_60.pt'
warm = r'$warm'
agent = PolicyOpponent(load_state_dict(e60), deterministic=True, name='e60')
opp = PolicyOpponent(load_state_dict(warm), deterministic=True, name='warm')
env = Bobble_league_env(opponent=opp, max_ep_len=200, step_timeout=30.0)
w=l=d=0
env.reset()
for _ in range($Games):
    done=False
    while not done:
        a=agent.act(env._raw_obs)
        try:
            o,r,term,trunc,info=env.step(a)
        except TimeoutError:
            d+=1; done=True; env.reset(); break
        if term or trunc:
            s=info.get('scorer',0) if term else 0
            w+=s==1; l+=s==2; d+=(s not in (1,2)); done=True; env.reset()
env.close()
n=max(1,w+l+d)
print(f'epoch60 vs warm_start: W{w} L{l} D{d}  win rate {w/n:.0%}')
"@ 2>&1 | Tee-Object -FilePath $outLog -Append
    Pop-Location
}
finally {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "e60eval" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
}

Write-Host ""
Write-Host "== epoch-60 re-test done; see $outLog and $eloCsv =="
exit 0
