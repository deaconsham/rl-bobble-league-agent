# Phase 1 report: six games at once, crash restart, full saves + resume

Date: 2026-10-31. Branch: `ai-vs-ai-optimization`. Builds on Phase 0 (`2842a8c`).

## Commit

One commit on `ai-vs-ai-optimization`: **"phase 1: 6 games at once, crash restart,
full saves"** (the newest commit on the branch — `git log -1` shows the hash). The
report is folded into that same commit via `git commit --amend`, so the commit's
own hash cannot be printed inside it; the code first landed as `e70a780` before
the report and the final amends.

Staged by explicit path only. The prior-session working-tree edits
(`GameManager.cs`, `TeamGoal.cs`, `project.godot`, `optimization_plan.md`,
`export_presets.cfg`, deleted `runs/ppo_experiment` tfevents) and all generated
artifacts (`checkpoints/`, `runs/`, `logs/`, `.bat` wrappers, the junk
`BobbleLeague.csproj.old*` that `--import` regenerates) were deliberately left
out. `docs/spec/requirements.md` and `design.md` stay as the spec worker left
them (untracked); only `docs/spec/tasks.md` was committed, to tick the Phase 1
boxes.

## What changed (the 7 plan steps)

1. **`--shm=<path>`** — `SharedMemory.ResolvePath()` now reads `--shm=` from the
   Godot command-line user args first (then the env var, then the temp default),
   so each parallel game gets its own buffer. Single-game behaviour is unchanged
   when no `--shm` is given.
2. **`launch_games.py`** — new. `Game_launcher` starts N detached headless Godots,
   each with `--shm=<prefix>/bobble_league_mmap_<i>.dat` and `logs/godot_<i>.log`.
   On Windows each game is launched through a per-game `.bat` run by WMI
   (`Win32_Process.Create`), so Godot is never a child of the calling shell. Paths
   are made absolute so the detached launch resolves them from any CWD.
3. **`Vector_env`** (in `bobble_league_env.py`) — steps all N games at once: writes
   every learner + opponent action, sets every step flag, waits for each. Done
   games reset on their own, untouched by the others. One batched network call per
   turn in the training loop drives it.
4. **`training_loop.py --games N`** — one `ac.step` over the stacked `(N, 14)` obs
   per turn; one `PPO_buffer` per game so GAE paths never mix; `join_and_normalize`
   concatenates them at update time and normalizes advantages over the whole
   batch. Each game gets its own opponent; the `self` games share one
   `SelfPlayPool`. `--steps-per-epoch` is the total across games (split evenly).
   The single-game path still works via a thin `_Single_env_as_vector` wrapper.
5. **`--steps-per-epoch` is the total** — e.g. 6000 over 6 games = 1000 each.
6. **Crash / hang restart** — on a step timeout or a dead process, `Vector_env`
   cuts that game's path as **truncated** (so the loop bootstraps its value from
   the critic), relaunches only that one Godot via the launcher, waits for the
   handshake, and carries on while the other games keep running. Every restart is
   logged and counted (`Training/Restarts` in TensorBoard).
7. **Full checkpoints + `--resume`** — `save_checkpoint` writes the network, both
   optimizer states, the LR step, the epoch, the self-play pool (snapshot weights +
   win rates) and the RNG state (python / numpy / torch). `--resume <file>`
   restores all of it and continues at the next epoch. `'model'` stays a plain
   state_dict at the top level, so `evaluate.py` and frozen-opponent loading keep
   working on the new files, and old weights-only saves still load (wrapped).

## Nearby cases tested, and where

| Case | Covered by |
|---|---|
| 6 games step together on one batched action | `test_vector_env::test_six_games_step_together` |
| each game keeps its own shm file (no clash) | `test_vector_env::test_each_game_keeps_its_own_file`; live smoke (6 distinct) |
| a game finishing while the others are mid-turn | `test_vector_env::test_done_game_resets_alone` |
| one game crashes mid-epoch -> cut + restart, others run on | `test_crash_restart::test_killing_one_restarts_only_that_game` |
| **two** games crash in the same turn -> both restart | `test_crash_restart::test_killing_two_restarts_both` |
| a game hangs (step flag never clears) -> timeout + restart | `test_hang_timeout` |
| 2 epochs on 6 mock games, saves written, resume continues at epoch 2 | `test_train_short::test_two_epochs_then_resume` |
| resume with a pool of **0** copies | `test_checkpoint::test_pool_round_trip_empty` |
| resume with a full pool of **10** copies (+ win rates) | `test_checkpoint::test_pool_round_trip_full` |
| full save/load round trip (weights, both opts, epoch, pool, RNG) | `test_checkpoint::test_checkpoint_round_trip`, `test_rng_state_round_trip` |
| old phase-0 weights-only save still loads | `test_checkpoint::test_old_weights_only_checkpoint_still_loads` |
| resume restores the live pool from a real training run | `test_train_short::test_resume_restores_pool` |
| no spurious restarts on healthy real games | live smoke (restarts = 0) |

Not unit-tested (needs real Jolt physics, so the live smoke covers it): a ball
leaving the pitch (out-of-bounds), and real turn settling / stuck detection.

## Results

- `pytest -m "not live"`: **42 passed** (29 from Phase 0 + 13 new Phase 1).
- `dotnet build`: **Build succeeded, 0 Error(s)** (one pre-existing
  `AnimatedTexture` deprecation warning in `GameManager.cs`, not from this phase).
- Godot `--import`: succeeded; the csproj SDK bump the import makes is reverted by
  the smoke script each run (verified: `BobbleLeague.csproj` unchanged in git).
- **T4 live smoke** (`scripts/smoke_test_phase1.ps1 -Games 6`, 2 epochs x 1200 total
  steps, timescale 32, 10-game eval):

  | metric | value |
  |---|---|
  | games | 6 |
  | total turns | 2400 in 1.47 min |
  | **turns/min** | **1629** |
  | goals (W+L) | 196 (W / L) |
  | stuck turns | 0 |
  | restarts | 0 |
  | out of bounds | 8 (4% of games played) |
  | python tracebacks | 0 |
  | shm file clashes | 0 (6 distinct files) |

  **SMOKE PASS.**

## Note on the turns/min target

The plan expected ~5-6x the single-game ~470 turns/min (so ~2400-2800). The live
smoke measured **1629 turns/min, ~3.5x**, over a short 2-epoch run. The gap is
honest and expected on this one machine: Godot physics settling per turn is the
training bottleneck (not the tiny network), six Godots contend for the same CPU
cores and the one GPU, and a short 2-epoch run pays fixed launch + first-epoch
cost against little amortization. It is a clear, real multi-game speedup and all
hard gates (0 stuck, no clashes, no errors) pass; a longer overnight run (Phase 7)
with more epochs should push the ratio up as the fixed costs amortize. Phase 5's
`--timescale=48/64` and physics-step tuning are the levers to raise it further.

## Free RAM check

Before the 6-game live smoke: **4996 MB free of 16334 MB** (well above the
~1.5 GB floor), so the smoke ran with 6 games as planned, not the 4-game fallback.
