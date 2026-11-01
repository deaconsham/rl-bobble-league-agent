# Phase 8 report: close off Phase 7

Date: 2026-11-01. Branch: `ai-vs-ai-optimization`. Not pushed.

Phase 8 closes off the Phase 7 overnight run: it let the final eval finish,
re-tested epoch 60 to set the bar for later phases, wrote the Phase 7 report,
found and closed the ball leak, cleaned up, and ran the full test step.

## Commits

| Commit | Message | What |
|---|---|---|
| `e23cef0` | phase 7: overnight run and report | Phase 7 (and Phase 0) report; single-game training fix + test; smoke-harness fixes; two run scripts; rename cleanup. |
| `c8c588f` | phase 7: close the ball leak | `continuous_cd` on the ball + containment cage; out-of-bounds speed log. |

Split into two because the ball-leak fix is a separate concern. Both staged by
explicit path; `git diff --cached` of each was read before committing. Nothing
pushed.

## Test numbers

| Gate | Result |
|---|---|
| `pytest -m "not live"` (T1 + T2) | 119 passed (was 118; +1 regression test) |
| `dotnet build` | 0 errors (1 pre-existing unrelated `AnimatedTexture` warning) |
| Godot `--import` | scene parses; csproj SDK bump 4.6.1 -> 4.6.0 reverted |
| T4 live smoke (`scripts/smoke_test.ps1`) | PASS: goals 9, stuck 0, out of bounds 0 |
| 200 live games (`scripts/verify_no_oob.ps1`) | 0 balls out of the pitch, 0 stuck |

## Epoch-60 bar

Re-tested `p7_run/checkpoints/ppo_epoch_60.pt` over 40 games each
(`scripts/eval_epoch60.ps1`, output `p7_run/elo_epoch60.csv`), anchored to
chaser 1000 / defender 950 / replay 850 / random 800:

**Epoch-60 Elo = 825.9.** This is the bar later phases must beat.

| Opponent | Win rate (40 games) |
|---|---|
| random | 90% (W36 L4) |
| defender | 60% (W24 L5 D11) |
| chaser | 32% (W13 L24 D3) |
| replay | 42% (W17 L15 D8) |
| warm_start.pt | 62% (W25 L14 D1) |

Epoch 60 beats the warm start (62%) but loses to the chaser (32%), so it is
still easy to beat. The earlier "1080" came from 10-game matches; 825.9 over 40
games is the reliable number.

## Ball leak: cause and fix

**Evidence.** The ball left the pitch 176 times across the Phase 7 game logs.
Breakdown of the exit spots (pitch is X +-10, Z +-18.5; out at |X|>11, |Z|>19.5
or Y<-2):

| Exit mode | Count |
|---|---|
| Fell through the floor (Y < -2) | 98 |
| Past a side wall (|X| > 11) | 65 |
| Behind an end wall (|Z| > 19) | 21 (7 within the goal mouth) |

Every exit was at low height (max Y 0.76). Nothing went over a wall - the ball
is already clamped to y <= 1.2 in `Ball.cs`, below the 1.39 wall top.

**Cause.** The ball RigidBody had no continuous collision detection, and the
arena collision is a single zero-thickness `ConcavePolygonShape3D` trimesh. A
fast ball moves far enough in one physics step to pass straight through that
one-sided surface near a corner or seam, then either drops out (no floor beyond
the arena) or drifts out the side.

**Fix** (`bobble_league/Scenes/scene.tscn`):
- `continuous_cd = true` on the `Ball` RigidBody - Jolt sweeps the ball's path
  each step, so a fast ball can no longer skip through a wall.
- A `Containment` `StaticBody3D` cage of solid thin boxes just outside the pitch
  but inside the out-of-bounds bound, so a tunnelling step is still caught:
  floor top at y = -1 (above the Y<-2 bound, below the pitch), side walls at
  X = +-10.5 (inner face +-10), end walls at Z = +-19 (inner face +-18.5). Goals
  are detected at world Z ~= +-16.5 and trigger a reset before the ball reaches
  the end wall, so the cage never changes normal play.
- `GameManager.cs`: the out-of-bounds log now prints the ball's speed and
  velocity, not just its position.

**Nearby cases tested** (the leak check drives real games, so both ends and both
goals are exercised: the learner attacks one goal, the opponent the other):
- **Both end walls / both goals** - in the 200-game check and the 60-game
  stochastic stress run both teams scored (stress run: team 1 = 31 goals,
  team 2 = 27), so shots and clears happened at and behind both ends.
- **Fast balls** - a stochastic run (`--stochastic`, opponent `random`) sprays
  wild, full-power shots; 60 games, 0 out of bounds, 0 stuck.
- **Slow balls** - the deterministic epoch-60 run mixes soft shots (power from
  0.2); 200 games, 0 out of bounds.
- **Floor fall-through** (the biggest Phase 7 mode) - the cage floor catches any
  ball that tunnels downward; 0 Y<-2 exits across all runs.

Total across all Phase 8 live runs (200 + 60 + T4 smoke + the 10-game sanity):
**0 balls out of the pitch.**

## Long runs stop only on `--epochs`

Phase 7's warm run was cut short at epoch 251 by `p7_run/stop_warm.ps1`, not by
the training loop. Confirmed in code that no run cuts itself short: `ppo_train`'s
only epoch bound is `for epoch in range(start_epoch, epochs)` (training_loop.py
line 650), where `epochs` comes from `--epochs`. The `break` at line 834 is the
per-pass KL early stop inside one update (ends SGD passes for that epoch, not the
run) - standard PPO, not a run cut-short. So from now on a long run ends only
when `--epochs` is reached or when asked; no helper script should kill it. See
`phase7.md` for the incident.

## Pre-existing bug found and fixed

The single-game training path (`--games 1`) crashed with
`AttributeError: '_Single_env_as_vector' object has no attribute 'reward_config'`
at `training_loop.py:635`. The shim that wraps one env for the vector loop did
not forward `reward_config`, which the shaping-fade loop reads and rewrites each
epoch. Phase 7 trained with `--games 4` (the vector path), so it never hit this.
Fixed by forwarding the attribute to the wrapped env; added a T2 regression test
(`test_single_game_shaping_fade`). This is why the T4 smoke, which trains on the
single-game path, now passes.

## What I did with each uncommitted file

| File(s) | Decision |
|---|---|
| `bobble_league/Scenes/scene.tscn` | Committed (`c8c588f`) - the ball-leak fix. |
| `bobble_league/Scripts/GameManager.cs` | Committed (`c8c588f`) - out-of-bounds speed log. |
| `bobble_league/Scripts/TeamGoal.cs` | Committed (`e23cef0`) - profane method call renamed to `Score`. |
| `bobble_league/project.godot` | Committed (`e23cef0`) - `BobbleFuck` -> `Bobble League` / `BobbleLeague`. |
| `export_presets.cfg` | Committed (`e23cef0`) - same rename (`BobbleFuck` export path -> `BobbleLeague`); part of the one cleanup, so finished rather than left half-done. |
| `bobble_model/training_loop.py` | Committed (`e23cef0`) - single-game `reward_config` fix. |
| `tests/mock/test_train_short.py` | Committed (`e23cef0`) - regression test for the above. |
| `scripts/smoke_test.ps1` | Committed (`e23cef0`) - eval newest-by-time checkpoint, count goals from the eval log. |
| `docs/phase_reports/phase0.md` | Committed (`e23cef0`) - it had been left uncommitted at Phase 0. |
| `bobble_model/runs/ppo_experiment/*.tfevents.*SurfacePro11*` (23 deleted) | **Reverted the deletion** (restored all 23). They are tracked binary TensorBoard logs from another machine committed in an early joke-commit. Removing committed history is a data decision outside this phase's scope, so I restored them and raised a question rather than commit the deletion. |
| `docs/optimization_plan.md` (modified) | Left untouched - the planning doc, owned by the planning chats, not this phase. |
| `docs/spec/requirements.md`, `docs/spec/design.md` (untracked) | Left untouched - earlier worker's spec docs, not this phase. |
| `bobble_model/p7_eval.py` (untracked) | Left as a run artifact; referenced by `phase7.md`. Not swept into the commit. |
| `bobble_league/BobbleLeague.csproj.old*` (6 untracked backups) | Deleted - csproj-bump backups, clutter. |
| `bobble_model/checkpoints/`, `old_runs/`, `p7_run/`, `p7_scratch/`, other `runs/*` (untracked) | Left in place - run outputs and saves, kept in their folders (Phase 14 rule). My own test run dirs (`runs/smoke`, `runs/dbg_smoke`) and test logs were deleted. |

## New reusable scripts

- `scripts/verify_no_oob.ps1` - play N games, count balls out of the pitch.
- `scripts/eval_epoch60.ps1` - the epoch-60 40-game Elo re-test.

Both launch their own detached Godot on a private shm and stop only that Godot.

## Open question for the conductor

The 23 deleted `runs/ppo_experiment/*.tfevents.*SurfacePro11*` TensorBoard files:
I restored them rather than commit their deletion. They are stale binary logs
from a different host, committed long ago in `30c21da`. My recommendation is to
delete them from git (they are not source and `runs/` should probably be
gitignored), but that is a data decision for a human, so it is left for the
conductor to confirm.
