# Phase 14: long run 2 — started

Branch `ai-vs-ai-optimization`, on top of the Phases 8-12 review (`817a501`).
This report records the START of long run 2 and the proof it is training. The run
is still going; its results and the T5 strength eval are a later `phase14.md`.

## What this run is

A fresh 200-epoch run from the champion `best_p11_ln.pt` (LayerNorm epoch 80,
σ ≈ 0.20), with the Phase 11 learning recipe and every Phase 12 feature left at
its default of off, so this reproduces the Phase 11 config. Live Elo is on with a
dedicated spare Godot; the decade-boundary deadlock fix (`e12b446`) is in, and the
first live-Elo check at epoch 0 completed with no hang.

- **Epochs:** 200. `--steps-per-epoch 6000` is the *total* across all 6 games
  (1000 each), so 200 × 6000 = **1.2M turns**, inside the plan's "~1-2M turns".
- **Games:** 6 at timescale 48. Free RAM was ~5.8 GB at launch and settled to
  ~3.9 GB with all 7 Godots up (~1.7 GB for the games), so 6 held comfortably —
  no drop to 5 or 4 was needed.
- **Start point:** `--init-checkpoint bobble_model/checkpoints/best_p11_ln.pt`.
  `--init-checkpoint` starts a fresh optimizer and LR schedule (right for a new
  long run) and resets `log_std` to `INIT_LOG_STD` (-1.6). The champion is already
  at -1.6 (σ ≈ 0.20), so that reset is a **no-op** and its tight aim is kept.
  Value-norm stats start fresh on the init path (stat restore is resume-only), so
  the critic re-learns the value scale over the first epochs — expected, and EV
  recovered by epoch 3 (see below).
- **No KL teacher.** Phase 11 used a KL pull toward the warm start because it
  started *from* a warm start. Long run 2 starts from the strong champion, so a KL
  pull toward a warm start would drag the strong policy backward — the same call
  Phase 12 made when it started from the champion. `--kl-warm-start` is omitted.

## Config (exactly what was launched)

Phase 11 recipe, kept:

- `--use-layernorm --weight-decay 1e-4`
- `--value-norm --n-passes 5 --n-passes-v 15` (critic gets 3× the policy passes)
- `--clip-ratio 0.15 --n-minibatches 4`
- `--shaping-fade --shaping-fade-fraction 0.5` (legacy distance shaping, faded
  over the first half — with the Phase 12 flags off, `_shaping` uses
  `legacy_shaping`)

Phase 12 flags, left at their defaults (off), so a plain launch reproduces
Phase 11: no `--input-norm`, `--random-start-prob 0.0`, no `--reward-scale`,
no `--potential-shaping`.

Live Elo:

- `--live-elo-shm <spare, shm index 10> --live-elo-freq 10 --live-elo-games 20
  --live-elo-sigma 0.1`
- `--best-path bobble_model/p14_run/checkpoints/best_p14.pt`
- `best.pt` is replaced only when a challenger clears the current best by more
  than the 25-Elo margin (`live_elo.BEST_MARGIN_ELO = 25`, default) AND wins the
  head-to-head — the Phase 11 live-gate-noise guard.

## Launch command

The run is launched detached (survives the launching turn) by
`bobble_model/p14_run/launch_run.ps1`, which first brings up the spare live-Elo
Godot on shm index 10, then runs the trainer via a BOM-less `.bat`
(`Invoke-CimMethod Win32_Process Create`). The trainer launches its own 6 games.
The exact trainer command (from `bobble_model/p14_run/logs/run_train.bat`):

```bat
cd /d "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_model"
set PYTHONUNBUFFERED=1
python training_loop.py --epochs 200 --steps-per-epoch 6000 --games 6 --launch ^
  --godot "C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe" --project "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_league" --timescale 48 --shm-prefix "C:\Users\Decon Sham\AppData\Local\Temp" ^
  --opponent self --init-checkpoint "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_model\checkpoints\best_p11_ln.pt" ^
  --use-layernorm --weight-decay 1e-4 ^
  --shaping-fade --log-dir "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_model\p14_run\runs\p14_t3" --save-freq 10 ^
  --value-norm --n-passes 5 --n-passes-v 15 --clip-ratio 0.15 --n-minibatches 4 ^
  --live-elo-shm "C:\Users\Decon Sham\AppData\Local\Temp\bobble_league_mmap_10.dat" --live-elo-freq 10 --live-elo-games 20 --live-elo-sigma 0.1 ^
  --best-path "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_model\p14_run\checkpoints\best_p14.pt" > "...\p14_run\logs\train.log" 2>&1
```

## Run folder and files

Run folder: `bobble_model/p14_run/`.

- `launch_run.ps1` — detached launch (spare Godot + trainer).
- `resume_run.ps1` — resume from the newest full checkpoint after a crash/restart.
- `logs/train.log` — the trainer's stdout/stderr (progress lives here).
- `logs/run_train.bat` — the exact generated launch `.bat`.
- `logs/godot_10.log` — the spare live-Elo Godot's log.
- `checkpoints/best_p14.pt` — the live-Elo `best.pt` for this run (isolated).
- `runs/p14_t3/` — TensorBoard event files.

Periodic full saves are written by the trainer to
`bobble_model/checkpoints/ppo_epoch_<N>.pt` every 10 epochs (relative to the
`bobble_model` CWD — the shared dir). The earlier runs' `ppo_epoch_*.pt` were
archived to `bobble_model/p12_run/archive/` before launch, so nothing from an
earlier run is overwritten. The named champions
(`best_p11_ln.pt`, `best_p12.pt`, …) are separate files and untouched.

## PIDs (at the time of writing)

- Trainer (real python running `training_loop.py`): **55992**.
- Godot processes: **14** — the 6 training games (shm `mmap_0`..`mmap_5`) and the
  1 spare live-Elo game (shm `mmap_10`), each as a console-wrapper + engine-child
  pair. Spare-Elo pair: 54592 / 41480. PIDs drift as games are relaunched, so the
  durable handle is the `--shm=...mmap_<i>.dat` token, not a PID.

## Proof it is training (early stats, through epoch 4)

Launched 07:37:52 EST. Device: **cuda** (GTX 1080 Ti). Weights loaded from
`best_p11_ln.pt`, `log_std` reset to -1.6 (no-op). The first live-Elo check
(epoch 0) completed cleanly — no hang, the deadlock fix holds.

| epoch | KL | ClipFrac | EV | Entropy (σ≈0.20) | passes | restarts | turn mean / p95 / max (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0466 | 0.533 | 0.080 | -1.084 | 1 (early-stop) | 0 | 3.54 / 5.78 / 8.25 |
| 1 | 0.0215 | 0.427 | 0.247 | -1.077 | 5 | 0 | 3.28 / 5.17 / 7.83 |
| 2 | 0.0118 | 0.240 | 0.311 | -1.081 | 5 | 0 | 3.66 / 5.80 / 8.53 |
| 3 | 0.0098 | 0.276 | 0.572 | -1.089 | 5 | 0 | 3.42 / 5.47 / 7.77 |
| 4 | 0.0087 | 0.243 | 0.063 | -1.094 | 5 | 0 | 3.62 / 5.87 / 7.68 |

- **Turns/min:** ~4800 at steady state (epochs 2→4 took ~150 s for 2 epochs,
  ~75 s/epoch; ~1585 turns/min over the whole first 11 min if you fold in the
  ~1.5 min startup and the epoch-0 Elo check). Both well above the 4-game
  ~1600-1970 baseline.
- **KL:** the epoch-0 spike to 0.0466 (early-stop at pass 1) is the known
  fresh-optimizer / fresh-value-norm transient — the same epoch-0 spike Phase 12
  saw from the champion init. It settled under `target_kl` (0.02) by epoch 1 and
  is ~0.009 by epoch 4.
- **σ (action spread):** entropy steady at ~-1.08 → -1.09, i.e. σ ≈ 0.20 held
  (tight aim kept, not pulled back up).
- **EV:** 0.08 → 0.57 by epoch 3 (already clearing the Phase 11 0.4 target in
  peaks); per-epoch EV is noisy as the fresh value-norm warms up.
- **Restarts: 0. Stuck turns: 0. Tracebacks: 0.** First live-Elo @ 0: mean 961.5
  (fresh init-eval baseline), saved as the first `best_p14.pt`.
- **RAM:** ~3.9 GB free with all 7 Godots up.

## Estimated finish

Steady ~75 s/epoch × 200 ≈ 2.5 h of compute, plus ~20 live-Elo boundaries (every
10 epochs) at ~1-2 min each ≈ 0.5 h. **≈ 3.0-3.5 h from launch**, i.e. roughly
**10:40-11:10 EST** today. The run stops only at `--epochs 200`; it is never cut
short for a budget.

## How to check progress

```powershell
Get-Content "C:\Users\Decon Sham\Documents\rl-bobble-league-agent\bobble_model\p14_run\logs\train.log" -Tail 20
# current epoch:
(Select-String -Path "...\p14_run\logs\train.log" -Pattern "^Epoch: \d+" | Select-Object -Last 1).Line
# done marker when it finishes:
Get-Content "...\p14_run\logs\train.done"   # TRAIN_DONE_0 on a clean exit
```

TensorBoard: point `--logdir` at `bobble_model/p14_run/runs/p14_t3`.

## How to stop

Stop the trainer first, then its Godots (so no game free-runs), by their unique
shm tokens — not by the WMI cmd pid:

```powershell
# trainer:
Stop-Process -Id 55992 -Force    # or find it: Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -match 'training_loop.py' }
# its games + the spare, by shm token:
Get-CimInstance Win32_Process -Filter "Name like '%Godot%'" |
  ? { $_.CommandLine -match 'mmap_(0|1|2|3|4|5|10)\.dat' } |
  % { Stop-Process -Id $_.ProcessId -Force }
```

## How to resume

After a crash, hang, or machine restart: make sure no `training_loop.py` and no
Phase 14 Godot are still alive (a half-dead run holds the shm files), then run
`bobble_model/p14_run/resume_run.ps1`. It relaunches the spare live-Elo Godot and
resumes the trainer from the newest `bobble_model/checkpoints/ppo_epoch_<N>.pt`
via `--resume` (which restores the optimizer, the LR-decay step, the value-norm
stats, the self-play pool and the RNG, so the LR keeps decaying toward epoch 200
instead of snapping back to the full rate).
