# Phase 14: long run 2. A fresh 200-epoch run from the champion best_p11_ln.pt
# (LayerNorm epoch 80, sigma ~0.2), using the Phase 11 recipe with every Phase 12
# feature at its default of off, so this reproduces the Phase 11 learning config.
#
# 200 epochs x 6000 steps/epoch = 1.2M turns, inside the plan's "~1-2M turns" for
# Phase 14 (steps-per-epoch is the TOTAL across all 6 games, i.e. 1000 each).
#
# Phase 11 recipe kept: --use-layernorm, --weight-decay 1e-4, --value-norm,
#   --n-passes 5, --n-passes-v 15, --clip-ratio 0.15, --n-minibatches 4,
#   --shaping-fade (legacy distance shaping, faded over the first half).
# Phase 12 flags left off (defaults): no --input-norm, --random-start-prob 0.0,
#   no --reward-scale, no --potential-shaping -> _shaping uses legacy_shaping.
# No KL teacher: we start from the strong champion, not a warm start, so a KL pull
#   toward a warm start would drag the strong policy backward (same call as Phase 12).
# --init-checkpoint resets log_std to INIT_LOG_STD (-1.6); the champion is already
#   -1.6, so this is a no-op and its tight aim is kept. value_norm starts fresh on
#   the init path (restore is resume-only) and the critic re-learns the scale.
#
# Live Elo ON with a dedicated spare Godot on shm index 10 (clear of training
#   games 0-5 and prior runs' 8/9), freq 10, 20 games, sigma 0.1. The decade-
#   boundary deadlock fix (e12b446) is in: 20s step timeout, 180s check budget,
#   spare-game relaunch on hang. best.pt is best_p14.pt with the 25-Elo margin
#   guard (live_elo.BEST_MARGIN_ELO = 25, default).
#
# Everything is launched DETACHED (Invoke-CimMethod Win32_Process Create on a
#   BOM-less .bat with absolute paths) so it outlives the launching turn. The run
#   stops only at --epochs 200 or on an explicit --resume-free manual stop.
#
# Restart / resume: re-run this script to relaunch the spare Godot, then use
#   resume_run.ps1 (written alongside) to resume the trainer from the newest full
#   checkpoint in bobble_model/p14_run/checkpoints via --resume.
$ErrorActionPreference = "Stop"
$godot = "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe"
$model = "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_model"
$project = "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_league"
$run = Join-Path $model "p14_run"
$logs = Join-Path $run "logs"
$ckpts = Join-Path $run "checkpoints"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
New-Item -ItemType Directory -Force -Path $ckpts | Out-Null

# ---- 1. spare live-Elo Godot on its own shm (index 10), detached ----
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

# ---- 2. the 200-epoch training run, detached (launches its own 6 games) ----
$trainLog = Join-Path $logs "train.log"
$trainBat = Join-Path $logs "run_train.bat"
$trainBody = @"
cd /d "$model"
set PYTHONUNBUFFERED=1
python training_loop.py --epochs 200 --steps-per-epoch 6000 --games 6 --launch ^
  --godot "$godot" --project "$project" --timescale 48 --shm-prefix "$realTemp" ^
  --opponent self --init-checkpoint "$model\checkpoints\best_p11_ln.pt" ^
  --use-layernorm --weight-decay 1e-4 ^
  --shaping-fade --log-dir "$run\runs\p14_t3" --save-freq 10 ^
  --value-norm --n-passes 5 --n-passes-v 15 --clip-ratio 0.15 --n-minibatches 4 ^
  --live-elo-shm "$eloShm" --live-elo-freq 10 --live-elo-games 20 --live-elo-sigma 0.1 ^
  --best-path "$ckpts\best_p14.pt" > "$trainLog" 2>&1
echo TRAIN_DONE_%ERRORLEVEL% > "$logs\train.done"
"@
[System.IO.File]::WriteAllText($trainBat, $trainBody, (New-Object System.Text.UTF8Encoding($false)))
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"$trainBat`"" } | Out-Null
Write-Host "Phase 14 long run 2 launched; log at $trainLog"
Write-Host "elo shm: $eloShm"
Write-Host "checkpoints: $ckpts"
