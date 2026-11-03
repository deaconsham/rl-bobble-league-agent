# Requirements: Bobble League RL optimization

Derived from `docs/optimization_plan.md` (2026-10-31). Requirements are grouped by
plan phase, in the plan's order (Phase 0 through Phase 8), followed by the testing
rules. Each requirement is a user story with acceptance criteria in WHEN/THEN form.
Every "pass" check in the plan appears as at least one acceptance criterion.

Spelling is Canadian English (colour, centre, defence, behaviour); `-ize` endings
are kept. Engine and library API names are left as they are.

Branch: all work is on `ai-vs-ai-optimization`. Each phase is its own commit and is
committed only once its tests pass.

---

## Phase 0: codebase clean-up

### R1: Consistent project naming
As a developer, I want the Godot folder, C# project and Python module named for
"Bobble League" so that every later change builds on the final names.

Acceptance criteria:
- WHEN the repository is checked out, THEN the Godot folder is `bobble_league/` (not `bobble_fuck/`).
- WHEN the C# project is inspected, THEN it is `BobbleLeague.csproj`, and `assembly_name` plus `config/name` in `project.godot` and the export path in `export_presets.cfg` refer to Bobble League.
- WHEN the Python package is inspected, THEN the environment module is `bobble_league_env.py`, its env class is renamed from `Bobble_fuck_env`, and every `import` and reference (including `bobble_fuck_env_helper` and `training_loop.py`'s `from bobble_fuck_env import ...`) uses the new name.
- WHEN the score method is called, THEN it is named `Score()` in `GameManager.cs` and `TeamGoal.cs` (not `SCOREEEE...()`).

### R2: Remove junk and crude content
As a developer, I want dead files and crude comments gone so that the repository reads cleanly.

Acceptance criteria:
- WHEN the repository is listed, THEN `BobbleFuck.csproj.old`, `.old.1`, the root `.DS_Store`, and the duplicate root `BallProperties.tres` are absent.
- WHEN `DataManager.cs` is read, THEN the ASCII art and the `godgod...` block and crude comments are removed, while the casual lowercase comment style is kept.

### R3: Canadian English and consistent style
As a developer, I want spelling and code style consistent so that the codebase is uniform.

Acceptance criteria:
- WHEN comments, docs and our own identifiers are read, THEN Canadian English spelling is used (`defense` -> `defence`, colour, centre), `-ize` endings are kept, and engine API names are left unchanged.
- WHEN C# is read, THEN it uses 4-space indentation, braces on the same line, and short lowercase comments.
- WHEN Python is read, THEN classes use `Like_this` names and docstrings follow the "Docstring for ..." form.

### R4: Test harness exists
As a developer, I want a `tests/` tree with the mock Godot and the first unit tests so that changes can be checked automatically.

Acceptance criteria:
- WHEN the repository is listed, THEN a `tests/` directory exists containing the mock Godot and the first unit tests (see R34-R36).
- WHEN `pytest` is listed as a dependency, THEN it appears in `requirements.txt`.

### R5: Phase 0 passes its tests before commit
As a developer, I want Phase 0 validated so that the rename does not break the build.

Acceptance criteria:
- WHEN the Phase 0 test gate runs, THEN the T1 and T2 tests pass.
- WHEN `dotnet build` runs, THEN it reports 0 errors.
- WHEN Godot runs `--import`, THEN it completes (and any `.csproj` SDK bump it makes is reverted).
- WHEN the T4 live smoke run runs, THEN goals are scored and there are 0 stuck turns.

---

## Phase 1: six games at once, crash-safe

### R6: Per-game shared-memory path
As a developer, I want each Godot to use its own shared-memory file so that parallel games do not clash.

Acceptance criteria:
- WHEN Godot is launched with `--shm=<path>`, THEN `SharedMemory.cs` reads that path from the command line and uses it instead of the default temp file.
- WHEN no `--shm` is given, THEN the existing default-path behaviour is kept.

### R7: Launcher for N headless games
As a developer, I want a launcher that starts N headless Godots so that many games run at once.

Acceptance criteria:
- WHEN `launch_games.py` is run with N, THEN it starts N detached headless Godots, each logging to `logs/godot_<i>.log` and each given `--shm=<temp>/bobble_league_mmap_<i>.dat`.

### R8: Vector environment steps all games together
As a developer, I want one environment that steps all N games so that one network call serves them all.

Acceptance criteria:
- WHEN `Vector_env` steps, THEN it writes every game's action, sets every step flag, and waits for all games.
- WHEN a game finishes (goal), THEN it resets the same way the single env does, independently of the others.

### R9: Training loop runs N games
As a developer, I want `training_loop.py --games N` so that the loop drives the vector env.

Acceptance criteria:
- WHEN `--games N` is passed, THEN one network call per turn serves all N games.
- WHEN trajectories are collected, THEN each game has its own buffer so GAE paths stay separate, the buffers are joined at update time, and advantages are normalized over the whole batch.
- WHEN opponents are assigned, THEN each game gets its own opponent and all share one snapshot list.
- WHEN `--steps-per-epoch` is set, THEN it is the total across games (e.g. 6000 total with 6 games = 1000 each).

### R10: Restart a crashed or hung game
As a developer, I want a dead or stuck game restarted so that an overnight run survives one crash.

Acceptance criteria:
- WHEN a game's process exits or its step times out, THEN `Vector_env` cuts that game's path (truncated, bootstrapping from the critic), relaunches only that one Godot, waits for the handshake, and carries on while the other games keep running.
- WHEN a restart happens, THEN it is logged.

### R11: Full checkpoints and resume
As a developer, I want checkpoints to hold all run state so that a crashed run resumes exactly.

Acceptance criteria:
- WHEN a checkpoint is saved, THEN it stores the network, both optimizers, the LR schedule step, the epoch, the self-play pool weights and their win rates, and the RNG state.
- WHEN `--resume <file>` is passed, THEN the run picks up from that saved state.

### R12: Phase 1 passes its tests before commit
Acceptance criteria:
- WHEN T1 runs, THEN a save/load round trip passes.
- WHEN T2 runs, THEN 6 mock games run, one is killed mid-epoch, and the run resumes from a save.
- WHEN T4 runs with 6 real games, THEN there are no stuck turns or file clashes and turns/min is about 5-6x the single-game rate (~470).

---

## Phase 2: new action shape (angle + power)

### R13: Angle-and-power action per player
As a developer, I want each player controlled by an aim angle and a power so that no action number is wasted.

Acceptance criteria:
- WHEN an action is decoded, THEN each player uses 1 angle number and 1 power number (6 numbers total, none wasted).
- WHEN the angle `a` in [-1, 1] is decoded, THEN it maps to `a * pi` in the team's mirrored frame, with the -1/+1 seam placed at "straight back toward my own goal".
- WHEN the power `p` in [-1, 1] is decoded, THEN it maps to `0.2 + 0.8 * (p + 1) / 2`, so power ranges from 20% to 100% strength.

### R14: Launch takes a power scale
As a developer, I want `Player.Launch` to accept a power scale so that shots vary in strength.

Acceptance criteria:
- WHEN `Player.Launch` is given a power scale, THEN it applies that scale instead of always normalizing to full power.

### R15: Opponents and human input use angle and power
As a developer, I want the scripted opponent and human input in the new format so that every path speaks the final action.

Acceptance criteria:
- WHEN `ChaserOpponent` and `evaluate.py --human` act, THEN they output the same angle-and-power format.
- WHEN a human drags, THEN drag length becomes shot power.

### R16: Protocol version bump
As a developer, I want the version byte bumped so that mismatched builds refuse each other.

Acceptance criteria:
- WHEN the protocol is inspected, THEN it is still 120 bytes and the version byte is bumped.
- WHEN an old Python meets a new Godot, or vice versa, THEN they refuse each other on the version mismatch.

### R17: Phase 2 passes its tests before commit
Acceptance criteria:
- WHEN T1 runs, THEN angle/power decode is checked and the mirror frame still round-trips.
- WHEN T4 live smoke runs, THEN it passes.
- WHEN power is checked, THEN a power of 0.2 moves a player clearly less than a power of 1.0.

---

## Phase 3: PPO fixes

### R18: Minibatch SGD
As a developer, I want minibatch updates so that each epoch learns more.

Acceptance criteria:
- WHEN the PPO update runs, THEN it shuffles the batch, splits it into minibatches of 256, and runs 10 passes, replacing the 80 full-batch steps.

### R19: Per-pass KL early stop
Acceptance criteria:
- WHEN the update runs, THEN KL early stopping is kept but evaluated per pass, with `target_kl` in the 0.02-0.03 range.

### R20: Entropy bonus
Acceptance criteria:
- WHEN the policy loss is computed, THEN it is `loss_pi - ent_coef * entropy` with `ent_coef = 0.005`.

### R21: Gradient clipping
Acceptance criteria:
- WHEN gradients are applied, THEN `clip_grad_norm_(params, 0.5)` is used for both networks.

### R22: Linear LR decay
Acceptance criteria:
- WHEN the run progresses, THEN the learning rate decays linearly to 0 over the run.

### R23: Orthogonal initialization
Acceptance criteria:
- WHEN the networks are built in `neural_network.py`, THEN weights use orthogonal init with gain sqrt(2) for hidden layers, 0.01 for the policy head, and 1.0 for the value head.

### R24: PPO diagnostics logged
Acceptance criteria:
- WHEN training logs to TensorBoard, THEN `approx_kl`, clip fraction and explained variance are recorded so each change can be checked.

### R25: Phase 3 passes its tests before commit
Acceptance criteria:
- WHEN T1 runs, THEN one update on fake data gives finite losses, KL and clip fraction in range, and clipped gradients.
- WHEN the T3 health check runs after a ~30-min run, THEN KL early stop no longer fires at about step 5, entropy falls steadily, and the vs-chaser win rate rises faster than the 200-epoch run did at the same turn count.

---

## Phase 4: warm start from the chaser

### R26: Record chaser-vs-chaser data
Acceptance criteria:
- WHEN `warm_start.py` runs, THEN it plays chaser vs chaser on the 6 games for about 20 minutes and records `(obs, action)` pairs.

### R27: Behaviour cloning and critic fit
Acceptance criteria:
- WHEN warm start trains, THEN the policy is trained to copy the recorded actions (behaviour cloning, mean-squared error on the mean) and the critic is trained on the game results.
- WHEN cloning finishes, THEN `log_std` is raised back to its starting value so that PPO still explores.

### R28: Train from a warm-start checkpoint
Acceptance criteria:
- WHEN `training_loop.py --init warm_start.pt` is run, THEN training starts from the warm-start checkpoint.

### R29: Phase 4 passes its tests before commit
Acceptance criteria:
- WHEN T1 runs, THEN the cloning loss drops on a fixed recorded set.
- WHEN T5 runs, THEN the warm-started net beats the random net vs chaser before any PPO, and PPO from it is never worse than PPO from scratch at the same turn count.

---

## Phase 5: more and better data per turn

### R30: Live-vs-live games in the mix
As a developer, I want some games to be the live learner against itself so that both-sides data has something to train on.

Acceptance criteria:
- WHEN opponents are drawn, THEN `SelfPlayPool` has a "live" slot and the mix is about 55% frozen copies, 30% live learner, and 15% chaser.

### R31: Train on both teams in live-vs-live games
Acceptance criteria:
- WHEN a game is live vs live, THEN Team B's transitions are stored too (using its mirrored view), roughly doubling data in those games.
- WHEN the opponent is a frozen copy, THEN its turns are never stored (off-policy).

### R32: Timescale and physics tuning
Acceptance criteria:
- WHEN one Godot does not max its CPU core, THEN `max_physics_steps_per_frame` is raised and `--timescale=48/64` is tried, with physics unchanged because ticks scale.

### R33: Measure turn length
Acceptance criteria:
- WHEN training runs, THEN turn length in game seconds is logged as a histogram, and if most turns end well under 20 s, `MaxTurnSeconds` or friction is lowered and the effect re-checked.

### R34: Phase 5 passes its tests before commit
Acceptance criteria:
- WHEN T1 runs, THEN the both-sides buffer keeps the two teams' paths apart and the opponent mix matches the set weights over 10,000 draws.
- WHEN T2 and T4 run, THEN they pass.
- WHEN training runs, THEN a turn-length histogram appears in TensorBoard.

---

## Phase 6: measure strength, pick opponents well

### R35: League Elo evaluation
Acceptance criteria:
- WHEN `league_eval.py` runs every 20 epochs, THEN it plays a round-robin of checkpoints, computes an Elo per checkpoint, and saves the result to CSV.

### R36: Held-out test opponents
Acceptance criteria:
- WHEN strength is measured, THEN a fixed held-out set never enters the training pool: a chaser with different settings, a "defender" bot that stays in front of its goal, a random bot, and a replay of the winning move.

### R37: Best-checkpoint gate
Acceptance criteria:
- WHEN a new checkpoint is evaluated, THEN it must beat the previous best in more than 55% of 100 games to become the new "best".

### R38: Prioritized fictitious self-play (PFSP)
Acceptance criteria:
- WHEN old copies are picked, THEN selection is weighted by how often the learner loses to them, weight `(1 - winrate)^2`, instead of uniform, while about 15% chaser and about 30% live games are kept.

### R39: Reward-shaping fade
Acceptance criteria:
- WHEN training passes its halfway point, THEN the per-turn shaping weights (`RewardConfig`) have faded to about 0, so the AI plays for goals rather than "ball moved forward".

### R40: Play-time randomness
Acceptance criteria:
- WHEN `evaluate.py --human` runs, THEN actions are sampled with std x 0.3 so that one fixed counter does not always work.

### R41: Phase 6 passes its tests before commit
Acceptance criteria:
- WHEN T1 runs, THEN Elo math on known results matches hand calculation, PFSP weights are checked, and the shaping weight hits 0 at the halfway epoch.
- WHEN T5 runs, THEN it runs on every 20-epoch save.

---

## Phase 7: overnight run

### R42: Resumable overnight run
Acceptance criteria:
- WHEN the overnight run runs, THEN it trains about 1-2M turns on 6 games, is `--resume` ready, and writes full saves every 10 epochs.

### R43: Phase 7 passes its tests before commit
Acceptance criteria:
- WHEN T5 runs at the end, THEN Elo rises across the run, the model beats the held-out set more than the 200-epoch run did, and the repeated move scores in under 20% of tries.

---

## Phase 8: only if it is still easy to beat

### R44: Exploiter policy
Acceptance criteria:
- WHEN about every 50 epochs pass, THEN a short-lived second policy is trained only against the frozen main AI, and it is added to the pool so the main AI learns to block its tricks.

### R45: Left-right mirror loss
Acceptance criteria:
- WHEN the mirror loss is applied, THEN it penalizes `pi(mirror(s)) != mirror(pi(s))` (the field is symmetric in X) rather than adding raw mirrored samples.

### R46: Teammate-shared network
Acceptance criteria:
- WHEN the shared net is used, THEN one small net per player sees itself, the ball, teammates and enemies with the same weights for all 3 players, so swapping teammate slots cannot confuse it and it learns once for all 3.

### R47: Phase 8 passes its tests before commit
Acceptance criteria:
- WHEN T1 runs for each piece, THEN the mirror of a mirror is the same board and swapping teammates swaps their actions.
- WHEN T5 runs, THEN the Phase 8 model is measured against the Phase 7 model.

---

## Testing rules

### R48: Test levels and gating
As a developer, I want every phase gated on tests so that no phase is committed broken.

Acceptance criteria:
- WHEN any phase is committed, THEN its tests pass first.
- WHEN a change is made, THEN fast tests (T1, T2) run on every change and slow tests (T3-T5) run at the end of a phase.
- WHEN `pytest -m "not live"` is run, THEN only T1 and T2 run; live tests are marked `live`.
- WHEN Python, NumPy and PyTorch are used in tests, THEN they are seeded so that T1 and T2 give the same result every run.
- WHEN Godot physics is involved (T3-T5), THEN tests pass on ranges, not exact numbers.
- WHEN a bug is found by hand, THEN a test is added before it is fixed.

### R49: T1 unit tests
Acceptance criteria (each file under `tests/unit/`):
- WHEN `test_protocol.py` runs, THEN every field's byte offset and size in the 120-byte layout, the version byte, and old-version refusal are checked.
- WHEN `test_frames.py` runs, THEN mirror-twice equals the same board, Team B's view of kickoff equals Team A's, and flipping the action back matches Godot's.
- WHEN `test_actions.py` runs, THEN angle/power decode is checked at the edges (-1, 0, 1) and power stays in [0.2, 1].
- WHEN `test_rewards.py` runs, THEN each `RewardConfig` term is checked on hand-made boards, the goal sign is checked for both teams, and the shaping fade schedule is checked.
- WHEN `test_gae.py` runs, THEN `discount_cumsum` and GAE match hand-worked numbers, a terminal gives 0 bootstrap, and a truncated path uses the critic.
- WHEN `test_ppo_update.py` runs, THEN one update on fake data gives finite losses, logs KL and clip fraction, keeps grad norm <= 0.5, and decays the LR.
- WHEN `test_checkpoint.py` runs, THEN save-then-load gives the same weights, optimizer state, pool, epoch and RNG.
- WHEN `test_opponents.py` runs, THEN the opponent mix over 10,000 draws, the PFSP weights, and the chaser output shape and range are checked.
- WHEN `test_elo.py` runs, THEN Elo after known results matches hand math.

### R50: T2 mock-game tests
Acceptance criteria (`tests/mock/`, using `tests/mock_godot.py`, a Python process that opens a shared-memory file, answers reset/step with simple fake physics, scores now and then, and can be told to hang or die):
- WHEN `test_single_game.py` runs, THEN reset -> steps -> goal ends the game -> reset again works.
- WHEN `test_vector_env.py` runs, THEN 6 mock games step together, each keeps its own file, and done games reset alone.
- WHEN `test_crash_restart.py` runs, THEN killing 1 of 6 mid-epoch lets the run go on, cuts that game's path, and restarts that game.
- WHEN `test_hang_timeout.py` runs, THEN a game that never clears its step flag times out and restarts.
- WHEN `test_train_short.py` runs, THEN 2 epochs of training on 6 mock games write saves and `--resume` continues at epoch 2.
- WHEN `test_evaluate.py` runs, THEN `evaluate.py` loads the save and plays a few games.

### R51: T3 training-health checks
Acceptance criteria (TensorBoard after about 30 min of real training):
- WHEN health is checked, THEN `approx_kl` is mostly under `target_kl` and early stop is not at about step 5.
- WHEN health is checked, THEN clip fraction is about 0.1-0.3.
- WHEN health is checked, THEN explained variance is above 0 and rising.
- WHEN health is checked, THEN entropy falls slowly (not flat, not crashing).
- WHEN health is checked, THEN turns/min is about 5-6x one game with 6 games.
- WHEN health is checked, THEN there are 0 stuck turns and restarts are rare and logged.

### R52: T4 live smoke test
Acceptance criteria (`scripts/smoke_test.ps1`):
- WHEN the smoke test runs, THEN `dotnet build` reports 0 errors.
- WHEN the smoke test runs, THEN Godot `--import` runs and any `.csproj` SDK bump it makes is reverted.
- WHEN the smoke test runs, THEN games launch detached (not as a child of the shell), 2 short epochs run, then `evaluate.py --opponent chaser --episodes 10` runs.
- WHEN the smoke test finishes, THEN it passes only if goals > 0, there are 0 stuck turns, out-of-bounds is at most 1 in 10 games, and there are no Python errors.
- WHEN the smoke test ends, THEN it stops the Godots it started, and only those.

### R53: T5 strength test
Acceptance criteria:
- WHEN the strength test runs, THEN it reports the Elo of the new save against earlier saves, the win rate against each held-out opponent over 100 games, and how often the repeated move still scores.


---

## Phases 9-16: final state (addendum)

R1-R53 above were written against the original Phase 0-8 plan. The plan was
renumbered and extended to Phase 16 (`docs/optimization_plan.md` is the source of
truth). These requirements record what the later phases actually shipped. Where a
value here differs from an earlier R (the Phase 3 entropy/clip defaults changed),
this addendum wins.

### R54: Shipped PPO defaults
As a developer, I want the trainer's defaults to match what was tuned so that a
plain launch reproduces the Phase 11 recipe.

Acceptance criteria:
- WHEN `training_loop.py` runs with no overrides, THEN `--ent-coef` is 0.0,
  `--target-kl` is 0.02, `--minibatch-size` is 256, `--n-passes` is 10, and
  `--max-grad-norm` is 0.5.
- WHEN the PPO clip is set, THEN it comes from `--clip-ratio` (default 0.2), not a
  hard-coded constant.

### R55: Live Elo during training
Acceptance criteria:
- WHEN `--live-elo-shm` is set, THEN the newest save is scored every
  `--live-elo-freq` epochs on one spare Godot against the chaser (1000) and
  defender (950) anchors, on `league_eval`'s scale, and logged to TensorBoard.
- WHEN `--live-elo-shm` is unset, THEN no live-Elo scoring runs.
- WHEN a spare live-Elo game wedges, THEN the short per-turn timeout and the
  overall check budget end the check instead of idling the run.

### R56: best.pt gate and rollback
Acceptance criteria:
- WHEN a save's live Elo clears the current best by more than the 25-Elo margin,
  THEN it becomes the new best, UNLESS a head-to-head vs the current best shows a
  clear one-on-one loss (score < 0.45).
- WHEN a save only ties or wins a noisy head-to-head but clears the margin, THEN
  it is still promoted (the head-to-head is a veto on a clear loss only, not a
  second gate).
- WHEN the live Elo drops more than 100 below the best for 2 consecutive checks,
  THEN the run rolls back to best.pt.

### R57: Run-dir saves
Acceptance criteria:
- WHEN `--best-path` and `--checkpoint-dir` are set, THEN best.pt and the periodic
  `ppo_epoch_*.pt` saves are written under the run's own directory, so two runs
  never overwrite each other's epoch saves.

### R58: LayerNorm, value-norm, input-norm
Acceptance criteria:
- WHEN `--use-layernorm` is set, THEN the hidden layers use LayerNorm and the save
  format carries the extra layer; a run using it starts from a warm start rebuilt
  with it on.
- WHEN a LayerNorm checkpoint is loaded, THEN `PolicyOpponent.load` and
  `SelfPlayPool.load_state` detect it from the state dict and rebuild the net to
  match, with no flag passed.
- WHEN `--value-norm` is set, THEN value targets are normalized with a running
  mean/std saved in the checkpoint.

### R59: Phase 12 data/reward flags kept, default off
Acceptance criteria:
- WHEN `training_loop.py` runs with no Phase 12 flags, THEN `--input-norm`,
  `--random-start-prob` (0.0), `--reward-scale` and `--potential-shaping` are all
  off, so the launch reproduces the Phase 11 recipe.
- WHEN `--random-start-prob` places a layout, THEN the ball is never inside a goal
  and no player is inside a wall.

### R60: Phases 13 and 15 skipped
Acceptance criteria:
- WHEN the plan is followed, THEN Phase 13 (bins) is skipped because aim is already
  fine (action sigma ~0.19), and Phase 15 (exploiter / mirror loss / shared net) is
  skipped because the AI is no longer easy to beat after long run 2.

### R61: Champion and default model
Acceptance criteria:
- WHEN Phase 14 finishes, THEN its epoch-100 save ships as
  `checkpoints/best_final.pt` and is the strongest model on the 40-game T5.
- WHEN `evaluate.py` runs with no `--checkpoint`, THEN it uses
  `checkpoints/best_final.pt`, falling back to the newest `ppo_epoch_*.pt` when the
  final model is absent.

### R62: Phase 16 clean-up, docs and final tests
Acceptance criteria:
- WHEN the repo is cleaned up, THEN the top-level `scripts/` holds only the live
  tools (`smoke_test.ps1`, `proof_live_elo.ps1`) and the superseded one-off
  per-phase scripts are under `scripts/archive/` with a README, kept for
  provenance.
- WHEN the docs are updated, THEN `README.md` and `docs/spec/` match the final
  code (new flags, run dirs, live Elo + best gate, LayerNorm, phases 13/15
  skipped).
- WHEN the final test gate runs, THEN the full `pytest` passes, `dotnet build`
  reports 0 errors, Godot `--import` runs (any `.csproj` SDK bump reverted), the
  T4 6-game smoke passes, and the T5 on `best_final.pt` vs the four anchors plus
  `best_p11_ln.pt` is recorded with the repeated-move exploit rate.
- WHEN human-vs-AI feel is checked, THEN it is done by hand (stop speed, 10 s turn
  limit, kick power) via `python evaluate.py --human` against `best_final.pt`.
