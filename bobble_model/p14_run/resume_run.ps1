# Phase 14 resume: bring the run back from the newest full checkpoint after a
# crash, a hang, or a machine restart. It does NOT change the config -- same
# recipe as launch_run.ps1 -- it only swaps --init-checkpoint for --resume, which
# restores the optimizer, LR-decay step, value-norm stats, self-play pool and RNG
# so training continues from where it stopped (and the LR keeps decaying toward
# epoch 200 instead of snapping back to the full rate).
#
# Periodic full saves are written by the trainer to
#   bobble_model/checkpoints/ppo_epoch_<N>.pt  (every --save-freq = 10 epochs),
# relative to the bobble_model CWD. This picks the highest-numbered one.
#
# Steps to resume:
#   1. Make sure no training_loop.py and no Phase 14 Godot are still running
#      (a half-dead run holds the shm files). Kill leftovers by their unique
#      --shm token if needed, not by the WMI-returned cmd pid.
#   2. Run this script. It relaunches the spare live-Elo Godot (shm index 10),
#      then relaunches the trainer with --resume pointing at the newest save.
$ErrorActionPreference = "Stop"
$godot = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe"
$model = "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_model"
$project = "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_league"
$run = Join-Path $model "p14_run"
$logs = Join-Path $run "logs"
$ckpts = Join-Path $run "checkpoints"

# newest full checkpoint (highest epoch number)
$latest = Get-ChildItem (Join-Path $model "checkpoints\ppo_epoch_*.pt") -ErrorAction SilentlyContinue |
  Sort-Object { [int]($_.BaseName -replace 'ppo_epoch_', '') } | Select-Object -Last 1
if (-not $latest) { Write-Host "FAIL: no ppo_epoch_*.pt to resume from in $model\checkpoints"; exit 1 }
Write-Host "resuming from $($latest.FullName)"

# ---- 1. spare live-Elo Godot on shm index 10, detached ----
$realTemp = "C:\Users\Decon Sham\AppData\Local\Temp"
$eloShm = Join-Path $realTemp "bobble_league_mmap_10.dat"
if (Test-Path $eloShm) { Remove-Item $eloShm -Force }
Push-Location $model
$env:PYTHONUNBUFFERED = "1"
python -c @"
from launch_games import Game_launcher, shm_path_for, log_path_for
l = Game_launcher(1, godot_exe=r'$godot', project_path=r'$project', shm_prefix=r'$realTemp', log_dir=r'$logs', timescale=48)
l.shm_paths = [shm_path_for(10, r'$realTemp')]
l.log_paths = [log_path_for(10, r'$logs')]
l.launch_one(0)
print('spare elo godot launched, shm', l.shm_paths[0])
"@
Pop-Location
$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path $eloShm) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (-not (Test-Path $eloShm)) { Write-Host "FAIL: spare elo godot never created shm"; exit 1 }
Write-Host "spare elo godot up on $eloShm"

# ---- 2. resume the training run, detached ----
$trainLog = Join-Path $logs "train_resume.log"
$trainBat = Join-Path $logs "run_train_resume.bat"
$trainBody = @"
cd /d "$model"
set PYTHONUNBUFFERED=1
python training_loop.py --epochs 200 --steps-per-epoch 6000 --games 6 --launch ^
  --godot "$godot" --project "$project" --timescale 48 --shm-prefix "$realTemp" ^
  --opponent self --resume "$($latest.FullName)" ^
  --use-layernorm --weight-decay 1e-4 ^
  --shaping-fade --log-dir "$run\runs\p14_t3" --save-freq 10 ^
  --value-norm --n-passes 5 --n-passes-v 15 --clip-ratio 0.15 --n-minibatches 4 ^
  --live-elo-shm "$eloShm" --live-elo-freq 10 --live-elo-games 20 --live-elo-sigma 0.1 ^
  --best-path "$ckpts\best_p14.pt" > "$trainLog" 2>&1
echo TRAIN_DONE_%ERRORLEVEL% > "$logs\train_resume.done"
"@
[System.IO.File]::WriteAllText($trainBat, $trainBody, (New-Object System.Text.UTF8Encoding($false)))
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"$trainBat`"" } | Out-Null
Write-Host "Phase 14 resume launched; log at $trainLog"
