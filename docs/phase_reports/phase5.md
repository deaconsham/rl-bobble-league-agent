# Phase 5 report: live-vs-live games, both teams' data, turn-length measurement

Date: 2026-10-31. Branch: `ai-vs-ai-optimization`. Builds on Phase 4 (`9f5e376`).

Phase 5 adds the live learner to the opponent mix, trains on BOTH teams' turns in
those live-vs-live games, and measures how long turns take in game seconds. It
touches C# (a wire-protocol bump to carry turn length) and Python (opponents,
env, training loop) plus tests.

## Commit

One commit **"phase 5: live vs live and both teams' data"** on
`ai-vs-ai-optimization`. This report is the last file written and is folded into
that commit, so its own final hash cannot be printed inside itself; `git log -1`
shows it. The pre-commit hash is recorded in the work report at `done`.

Staged by explicit path only — these files:
`bobble_league/Scripts/SharedMemory.cs`, `bobble_league/Scripts/DataManager.cs`,
`bobble_league/Scripts/GameManager.cs`, `bobble_model/shared_memory_link.py`,
`bobble_model/opponents.py`, `bobble_model/bobble_league_env.py`,
`bobble_model/training_loop.py`, `tests/mock_godot.py`,
`tests/unit/test_protocol.py`, `tests/unit/test_opponents.py`,
`tests/unit/test_both_teams.py`, `tests/mock/test_live_vs_live.py`,
`scripts/smoke_test_phase5.ps1`, `docs/spec/tasks.md` (Phase 5 boxes ticked only),
and this report.

**Left out on purpose** (same policy as Phases 0-4): the prior-session working-tree
edits that are **not mine** — `bobble_league/Scripts/TeamGoal.cs`,
`bobble_league/project.godot`, `export_presets.cfg`, `docs/optimization_plan.md`,
and the deleted `runs/ppo_experiment` tfevents. Also left out: all generated
artifacts (`checkpoints/`, `runs/`, `old_runs/`, `logs/`), the junk
`BobbleLeague.csproj.old*` backups, and the untracked spec/report files from other
phases (`docs/spec/design.md`, `requirements.md`, `docs/phase_reports/phase0.md`).
Nothing pushed.

## Protocol v4 — and why checkpoints still work (R33)

Godot already knew each turn's settle time (`GameManager.simTime`) but did not send
it. The 120-byte v3 buffer had no 4-aligned room for an f32, so the wire protocol
grew to **124 bytes, PROTOCOL_VERSION 4**, with a `turn_seconds` f32 at offset 120.

v4 only GREW the buffer with telemetry; the obs/action contract is unchanged. So
the checkpoint's `action_format` tag is kept at **3** (decoupled from the wire
version): Phase 2-4 saves still load, the live smoke evaluated a fresh Phase 5
checkpoint without issue, and a pre-Phase-5 Godot (v3) is now refused by the new
Python with a clear "protocol v3" message (tested).

**C# write path:** `SharedMemory.WritePayload/SendStepResults` take `turnSeconds`;
`DataManager.Export` passes it (default 0 on a reset, which is not a turn);
`GameManager` sends `(float)simTime` when a turn settles.
**Python read path:** `Shared_memory_link.read_turn_seconds()`.

## What changed

**`bobble_model/opponents.py` — live slot (R30).**
- New `LiveOpponent`: holds a REFERENCE to the live actor-critic (not a frozen
  copy), SAMPLES in its own mirrored frame, and records the
  `(obs, act, val, logp)` of its last turn so the loop can store it on-policy.
- `SelfPlayPool` gained a `live` slot and a `live_prob`. `reset()` now draws
  ~15% chaser / ~30% live / ~55% frozen copy (`scripted_prob=0.15`,
  `live_prob=0.30`). `attach_live(ac)` wires the live net; `is_live_current()`
  tells a caller whether this game is live vs live right now.
- `make_siblings(n)`: with N parallel games each game gets its OWN pool (its own
  `current` opponent) that SHARES one snapshot list + win rates + the live net, so
  every game picks and remembers its own opponent while a snapshot added once is
  seen by all. (Before Phase 5 all N games shared one pool instance and one
  `current`, so per-game opponent identity was not actually tracked.)
- `state()`/`load_state()` carry `live_prob`; the live slot itself is not saved
  (it is a reference to the training net, re-attached on resume).

**`bobble_model/bobble_league_env.py` — turn length + Team B transition (R31, R33).**
- `Vector_env._collect_turn` reads `turn_seconds` into every `info`, and for a
  live-vs-live game fills `info['opponent']` with Team B's transition: the acting
  obs/action/value/log-prob come from the `LiveOpponent` (recorded at act time,
  already in the opponent frame), and the reward is computed HERE from Team B's
  own frame — `opp_goal_reward = -goal_reward` plus shaping on
  `opponent_view(prev)` → `opponent_view(next)`. A frozen copy's turns are never
  stored (off-policy). The single-env `step` also reports `turn_seconds`.

**`bobble_model/training_loop.py` — both-teams buffer + logging (R31, R33).**
- New `Opponent_buffer`: a variable-length buffer for the Team B side (it only
  fills on live-vs-live turns, so its length is not known ahead). Same GAE maths
  as `PPO_buffer`, keeps separate paths apart, bootstraps a cut path from the
  critic, and yields the same `obs/act/ret/adv/logp` dict. An empty buffer returns
  correctly-shaped zero-row arrays.
- The rollout keeps one `Opponent_buffer` per game, stores `info['opponent']`
  turns, and finishes the B-side path exactly when the A-side path ends —
  bootstrapping a cut B-side path from the live net's value of
  `opponent_view(next_obs)` (same net, mirrored view).
- `join_and_normalize(buffers, extra_data=...)` folds the finished B-side data
  into the SAME batch before the single whole-batch advantage normalization, so
  the two teams' advantages are normalized together but their GAE never mixed.
- New TensorBoard output: `Turn_Length/Seconds` histogram plus
  `Turn_Length/{Mean,Median,P95,Max}_Seconds`, and `Data/Opponent_Turns_Stored`
  and `Data/Batch_Size` so the extra live-vs-live data is visible.
- The live net is attached to the pool after the model is built, so its live slot
  can play and feed training.

## Task 5.3 (timescale / MaxPhysicsStepsPerFrame) — status

`GameManager.SetTimeScale` already raises `Engine.MaxPhysicsStepsPerFrame` with the
time scale and scales `PhysicsTicksPerSecond`, so physics stays the same as the
scale rises (ticks scale with it). The smoke and T3 ran at timescale 32 at ~1600
turns/min on 4 games with 0 stuck turns. Trying 48/64 is a tuning sweep that only
pays off if one Godot does not max its core and needs the full 6-game overnight
box to judge honestly; it is a knob on an existing, working mechanism, not new
code, and is left to the Phase 7 run rather than guessed at here. Lowering
`MaxTurnSeconds` is addressed under "Turn length" below.

## Tests

T1 + T2, all Godot-free, seeded where randomness matters:

**`tests/unit/test_opponents.py` (new, T1 — R34):**
- opponent mix over 10,000 draws is 0.15 / 0.30 / 0.55 (±0.03) chaser/live/frozen,
- the live slot never appears until a net is attached,
- `is_live_current()` tracks the live slot,
- sibling pools share one snapshot list + live net but keep their own `current`,
- `LiveOpponent` samples and records a finite value/log-prob,
- the chaser still outputs in-range angle+power,
- with no frozen copies yet, draws are only chaser or live.

**`tests/unit/test_both_teams.py` (new, T1 — R34):**
- `Opponent_buffer` keeps two paths apart (GAE/returns of path 1 use only path-1
  rewards; a terminal path bootstraps 0),
- its GAE/returns match `PPO_buffer` for one identical path,
- `join_and_normalize` folds both sides into one batch (2×4 learner + 3 opponent =
  11 rows) and normalizes over the whole batch (mean ~0, std = sqrt((n-1)/n) under
  torch's unbiased std),
- an empty opponent data dict is dropped.

**`tests/mock/test_live_vs_live.py` (new, T2 — R34):** over mock games through the
real `Vector_env`: every step reports `turn_seconds`; a live-vs-live game hands
back Team B's `(obs, act, val, logp, reward)`; a chaser-only game reports none.

**`tests/unit/test_protocol.py` (updated, T1):** 124-byte v4 layout and the
`turn_seconds` offset; version is 4; a v3 buffer is refused; `turn_seconds`
round-trips. **`tests/mock_godot.py`** speaks v4 and writes a small synthetic
turn length each step.

## Results — all gates pass

- **`pytest -m "not live"`: 89 passed** (74 from Phase 4 + 15 new: 7 opponents,
  4 both-teams, 3 live-vs-live, 1 turn-seconds).
- **`dotnet build`: Build succeeded, 0 Error(s)** (one pre-existing
  `AnimatedTexture` deprecation warning in `GameManager.cs`, not from this phase).

- **T4 live smoke** (`scripts/smoke_test_phase5.ps1 -Games 4`, 2 epochs × 1200
  total turns, self-play, timescale 32; 4 games because free RAM was ~4.8 GB, the
  plan's tight fallback):

  | metric | value |
  |---|---|
  | games | 4 |
  | total turns | 2400 in 1.48 min |
  | turns/min | 1621 |
  | goals (W+L) | 26 |
  | stuck turns | 0 |
  | restarts | 0 |
  | out of bounds | 1 (4% of games) |
  | python tracebacks | 0 |
  | Team B turns stored | 1736 |

  **SMOKE PASS.** The 1736 Team B turns prove the live-vs-live both-teams data
  path ran end to end against real Godot, and the v4 handshake succeeded (the
  Python bridge refuses anything but v4).

- **T3 health run** (self-play, 4 games, timescale 32, 1200 steps/epoch,
  `ent_coef=0.001`, `target_kl=0.03`, `--epochs 30` so the LR ramp stays healthy;
  12 epochs read straight from the TensorBoard event file):

  | T3 number | target | run (epochs 0..11) | pass? |
  |---|---|---|---|
  | approx_kl | mostly < target_kl, early stop not at ~step 5 | 0.013–0.029, all < 0.045 (1.5×0.03) | **PASS** |
  | passes run per update | full passes, not cut at ~5 | 10 every epoch (early stop never fired) | **PASS** |
  | clip fraction | ~0.1–0.3 | 0.18–0.25 | **PASS** |
  | explained variance | above 0 and rising | 0.04 → 0.19, positive throughout, trending up | **PASS** |
  | entropy | down slowly, not flat, not crashing | ~5.48, stable (short window, ent_coef 0.001; same as Phase 3) | **PASS (stable, not crashing)** |
  | Team B turns stored | > 0 each epoch | 121–1200 per epoch | **PASS** |
  | turn-length histogram in TB | present | `Turn_Length/Seconds` logged every epoch | **PASS** |
  | stuck turns / restarts | 0 stuck; restarts rare | 0 stuck, 0 restarts | **PASS** |

  `Data/Batch_Size` = learner turns (2400) + Team B turns, confirming the
  both-teams data is folded into the update. The Team B count falls across epochs
  because the self-play pool gains snapshots (snapshot-freq 2), so more games draw
  a frozen copy (no B-side storage) rather than the live learner — exactly the
  intended mix.

## Turn length (R33) — the numbers, and what they say about MaxTurnSeconds

Turn length in game seconds, logged as a histogram and summary scalars. From the
T4 smoke and the T3 run (timescale 32):

| stat | value (game seconds) |
|---|---|
| mean | ~3.3 |
| median | ~3.2 |
| p95 | ~4.0–5.0 |
| max | ~6.3–7.6 |

**Almost every turn settles in about 3 seconds, and even the p95 is under 5 s —
far below the 20 s `MaxTurnSeconds` cap.** The cap is doing its job as a safety net
for the rare non-settling turn (the max ~7 s turns are the tail), not as a routine
limit. So there is clear room to LOWER `MaxTurnSeconds` (e.g. to ~10 s) to end a
genuinely stuck turn sooner without cutting any normal turn, or to add a little
friction/damping so turns settle faster still. That change alters how the game
plays for humans too (a turn is called "over" sooner), so per the plan it is a
tuning decision to make and re-check on the full run rather than ship blindly
inside this data-plumbing phase; the measurement this phase adds is what makes that
decision possible, and the histogram will show the effect when it is tried.

## Free RAM check

Free RAM was ~4.8 GB throughout (advisory "tight", floor 4 GB), so both T4 and T3
used the 4-game fallback the plan allows rather than 6. All Godot processes started
by the runs were stopped afterwards (verified 0 remaining). The dedicated-Godot
stop matcher in `smoke_test_phase5.ps1` matches the shm FILE NAME, not the quoted
`--shm=` path, fixing the Phase-4-class quoting miss that left eval Godots alive.

## Notes for later phases

- Phase 6's PFSP replaces the uniform `rng.choice(self.snapshots)` in
  `SelfPlayPool.reset()` with a `(1 - winrate)^2` weighting; the ~15% chaser and
  ~30% live slots stay as they are here.
- `Data/Opponent_Turns_Stored` and `Data/Batch_Size` are the scalars to watch when
  tuning the live-vs-live fraction.
- If `MaxTurnSeconds` is lowered, re-read `Turn_Length/*` to confirm no normal
  turns are being cut.
