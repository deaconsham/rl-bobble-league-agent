# Phase 9 report: train faster

Date: 2026-11-01. Branch: `ai-vs-ai-optimization`. Not pushed.

Phase 9 makes every later phase's test runs cheaper: 6 games instead of 4, a
10 s turn limit, a time scale above 32x, async stepping (no waiting on the
slowest game), parallel Elo, and a training-only stop speed that keeps human
play unchanged.

## What changed

| Step | Change | Where |
|---|---|---|
| 1 | 6 games, `--steps-per-epoch 6000` (run invocation; defaults left conservative) | `training_loop.py` CLI |
| 2 | `MaxTurnSeconds` 20 -> 10 | `GameManager.cs` |
| 3 | Time-scale editor hint 1,32 -> 1,128; default scale 48 | `GameManager.cs`, `training_loop.py`, `launch_games.py` |
| 4 | Async stepping in `Vector_env.step` | `bobble_league_env.py`, `shared_memory_link.py` |
| 5 | Elo over several games at once | `league_eval.py` |
| 6 | Optional Jolt / 30 Hz | NOT done (see below) |
| 7 | Faster stop (1.0) only in `ai_vs_ai`, 0.3 for human | `GameManager.cs`, `Ball.cs`, `Player.cs` |
| 8 | Optional multi-arena | NOT done (RAM is fine) |

### Step 4 — async stepping (the core change)

`Vector_env.step` used to write every game's action then `wait_for_step` on each
game in index order, so a fast game sat idle behind a slow one. It now writes all
actions, then polls the games and collects each the moment its turn is done
(`Shared_memory_link.step_ready()`, a non-blocking read of the step flag).

- Each game's result still goes into its own slot `i`, so a game's path stays in
  order and exactly one result is produced per game per step — no turn is lost or
  reordered.
- A game that hangs or whose process died is cut (`truncated`, `restarted=True`)
  once its per-game deadline passes and relaunched, exactly as before; the others
  keep going. `_collect_turn` (blocking wait + restart) became `_read_turn`
  (read only); the wait/timeout/restart now lives in `step`'s poll loop.

### Step 7 — training-only stop speed

The "stopped" speed (`speedThreshold`) was 1.0 in both scene files, which also
sped up human turns. `GameManager` now owns `TrainingStopSpeed = 1.0` and
`HumanStopSpeed = 0.3` and pushes the mode's value onto the ball (once) and every
player (in `Spawn`, since players are re-instantiated each turn) via a new
`SetStopSpeed` on `Ball`/`Player`. The scene's own 1.0 is only a fallback now.
`ai_vs_ai` keeps 1.0 (fast turns); any human mode gets 0.3 (nicer play).

### Steps 6 and 8 — not done, on purpose

- **Step 6 (Jolt solver steps / 30 Hz physics):** marked optional ("try only if
  time allows"). It changes how bounces play out and therefore human play, and
  the plan requires a 200-game check that goals/game, turn length and
  out-of-bounds all stay within ~10% before it may be kept. Async stepping plus
  time scale 48 already gave the speed-up the phase targets, so this was not
  attempted and nothing was kept. It remains available for a later pass.
- **Step 8 (several arenas in one Godot scene):** the plan says skip unless RAM
  forces it. Free RAM was 6.2 GB with 6 games running (~1 GB used), so it was not
  needed.

## Time scale: measured turns/min on 6 games

Measured with `scripts/archive/smoke_test_phase9.ps1` (6 games, `--launch`, 2 epochs x
1200 steps = 2400 turns, eval 10 games vs chaser). Turns/min is total game-turns
over the training wall clock (one training step plays one turn on every game).

| Time scale | turns/min | stuck | did-not-settle | out of bounds |
|---|---|---|---|---|
| 32 (old default) | 1760 | 0 | 0.00% | 0 |
| 48 (**chosen**) | 2346 | 0 | 0.00% | 0 |
| 64 | 2399 (2681 over 3 epochs, JIT warm) | 0 | 0.00% | 0 |

All three clear the pass bar: 0 stuck turns, did-not-settle well under 1%, and
turns/min above the 4-game ~1600-1970 range.

**Chosen scale: 48.** Turns/min rose sharply 32 -> 48 (+33%) but only marginally
48 -> 64 (+2%), so 48 captures ~98% of the throughput. At 64 each Godot runs at
~1.3-2.0 cores (Jolt is multithreaded, so no single Godot maxes one core), but
6 games push the host to ~9 of 12 cores, leaving little headroom for the training
process and the spare live-Elo game later phases add. 48 is the highest scale
where the rise is still real and the host keeps headroom.

## Step 5 — parallel Elo

`league_eval.run_league` now takes `--shms` (several running Godot games) and
spreads the round-robin pairings over them: `play_pairings_parallel` keeps up to
`len(games)` matches in flight on their own envs, interleaving their turns so a
fast match never waits on a slow one and a freed game picks up the next pairing.
A single `--shm` keeps the original one-game serial path, so existing callers are
unchanged. Spread over 4 games, a round-robin finishes in roughly a quarter of
the time.

## Nearby cases tested (async stepping)

`tests/mock/test_async_step.py` (4 new T2 tests, mock games with different turn
lengths via a new `step_delay` on `Mock_godot`):

- **Very different turn lengths** — one slow game (0.3 s) among instant ones:
  all four collected within one step (not 4x the slow time), each advanced
  exactly one turn.
- **No turn lost or reordered** — over 5 steps with four different turn lengths,
  every game ticked exactly 5 turns.
- **A game crashes mid-turn** — all games slow (real mid-turn window), kill one:
  it is cut (`truncated`, `restarted`) and relaunched, the other three untouched,
  the run carries on.
- **A game resets while others are mid-turn** — one game scores on turn 1 and
  resets while three slow games are still mid-turn; the reset does not disturb
  them (each still advances exactly one turn).

`tests/mock/test_league_parallel.py` (2 new T2 tests): the parallel Elo scheduler
plays every pairing exactly once over 4 games, and refills freed slots when there
are more pairings than games (6 pairings over 2 games).

## Test results

| Gate | Result |
|---|---|
| `pytest -m "not live"` (T1 + T2) | 125 passed (was 119; +4 async + 2 parallel-Elo) |
| `dotnet build` | 0 errors (1 pre-existing unrelated `AnimatedTexture` warning) |
| Godot `--import` | scene parses; csproj SDK bump 4.6.1 -> 4.6.0 reverted |
| T4 live smoke, 6 games @ ts 32 | PASS: 1760 turns/min, 0 stuck, 0.00% did-not-settle, 0 oob |
| T4 live smoke, 6 games @ ts 48 | PASS: 2346 turns/min, 0 stuck, 0.00% did-not-settle, 0 oob |
| T4 live smoke, 6 games @ ts 64 | PASS: 2399 turns/min, 0 stuck, 0.00% did-not-settle, 0 oob |

Turn lengths during the 6-game bursts: mean ~3.5 s, p95 ~5.8 s, max ~7.7 s — all
under the new 10 s limit, so no settle-time turn was cut.

## RAM seen

Free RAM 6.2 GB of 16 GB, ~1 GB used with 6 games running. Comfortable, so
6 games (not the 5-game fallback) and no multi-arena.

## New / changed files

- `bobble_league/Scripts/GameManager.cs` — turn limit 10 s, time-scale hint
  1,128, mode-dependent stop speed (`ApplyStopSpeed`, applied in `_Ready` and
  per player in `Spawn`).
- `bobble_league/Scripts/Ball.cs`, `Player.cs` — `SetStopSpeed`.
- `bobble_model/shared_memory_link.py` — `step_ready()` non-blocking poll.
- `bobble_model/bobble_league_env.py` — async `Vector_env.step`; `_collect_turn`
  -> `_read_turn`.
- `bobble_model/league_eval.py` — `--shms`, `play_pairings_parallel`, `_Match_slot`.
- `bobble_model/training_loop.py`, `launch_games.py` — default time scale 48.
- `tests/mock_godot.py` — per-game `step_delay`.
- `tests/mock/test_async_step.py`, `tests/mock/test_league_parallel.py` — new T2 tests.
- `scripts/archive/smoke_test_phase9.ps1` — the 6-game T4 at one time scale (launches its
  own detached games via `training_loop --launch`, stops only what it started).

## Recommended training invocation

    python training_loop.py --games 6 --steps-per-epoch 6000 --launch --timescale 48 ...

The `--games` and `--steps-per-epoch` defaults were left at 1 / 4000 so the
single-game path and the mock tests are unchanged; the long runs pass the 6-game
values explicitly, as earlier phases did.

## Commit

Staged by explicit path; `git diff --cached` of each tracked file was read before
committing. `docs/optimization_plan.md` (modified) was left out — it is the
planning doc, owned by the planning chats, not this phase. The untracked spec
docs, run folders and checkpoints were left untouched. Nothing pushed.

Commit: `phase 9: train faster`.
