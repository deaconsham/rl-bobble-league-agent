# Phase 4 report: warm start from the chaser (behaviour cloning before PPO) + T5

Date: 2026-10-31. Branch: `ai-vs-ai-optimization`. Builds on Phase 3 (`7219f93`).

Phase 4 is Python-only: a new `warm_start.py`, its unit tests and a T5 live
script. No Godot / C# code changed, so the action shape and protocol v3 are
untouched and Phase 4 checkpoints stay compatible with Phase 2/3 saves
(`action_format` unchanged). `training_loop.py --init-checkpoint` already existed
from Phase 1 and already loads the full warm checkpoint, so task 4.3 needed no
training-loop edit — it is covered by a test instead.

## Commit

One commit **"phase 4: warm start from chaser"** on `ai-vs-ai-optimization`. This
report is the last file written and is folded into that commit, so its own final
hash cannot be printed inside itself; `git log -1` shows it. The pre-commit hash
is recorded in the work report at `done`.

Staged by explicit path only — four files:
`bobble_model/warm_start.py`, `tests/unit/test_warm_start.py`,
`scripts/t5_warm_start.ps1`, `docs/spec/tasks.md` (Phase 4 boxes ticked only).

**Left out on purpose** (same policy as Phases 0-3): the prior-session
working-tree edits that are **not mine** — `bobble_league/Scripts/TeamGoal.cs`,
`bobble_league/project.godot`, `export_presets.cfg`, `docs/optimization_plan.md`,
and the deleted `runs/ppo_experiment` tfevents. Also left out: all generated
artifacts (`checkpoints/`, `runs/`, `old_runs/`, `logs/`), the junk
`BobbleLeague.csproj.old*` backups, and the untracked spec/report files from
other phases (`docs/spec/design.md`, `requirements.md`,
`docs/phase_reports/phase0.md`). Nothing pushed.

## What changed

**`bobble_model/warm_start.py` (new) — R26, R27.**

- `collect_demos(vec_env, chaser_factory, n_turns, ...)` (R26): plays the N games
  with a chaser on BOTH sides and records what the chaser does from the LEARNER's
  point of view. The learner slot is driven here by one chaser per game, fed the
  learner-frame raw obs, so each recorded action is already in the frame the
  network trains in (no frame conversion needed). One `PPO_buffer` per game gives
  GAE returns for the critic, paths kept apart exactly like training. Returns
  numpy `obs` (M,14), `act` (M,6), `ret` (M,).
- `behaviour_clone(ac, demos, ...)` (R27): minibatch supervised cloning — MSE of
  the policy MEAN (`pi_net`) against the recorded actions, and MSE of the critic
  (`v_net`) against the recorded returns. After cloning, `log_std` is reset to its
  init value `INIT_LOG_STD = -0.5` (matching `Actor_critic_neural_network`), so
  PPO still explores. Returns the per-epoch `(pi_loss, v_loss)` history so a test
  can check the loss drops.
- `save_warm_checkpoint(path, ac, ...)`: writes the FULL checkpoint format
  (Phase 1) with fresh optimizers, `epoch=0`, `lr_step=0` and the `action_format`
  tag, so `training_loop.py --init <path>` loads it and seeds the self-play pool.
- `warm_start(vec_env, out_path, ...)`: end to end (record -> clone -> reset
  log_std -> save). **The demo chaser driving the learner defaults to
  DETERMINISTIC** (`noise=0`, `kickoff_noise=0`) — see the diagnosis below. The
  env's own opponents stay the normal noisy chaser, so recorded STATES still vary.
- CLI `main()`: `--games/--turns/--clone-epochs/--timescale/...` records + clones +
  saves against N real Godot games (launched detached, as Phase 1), and
  `--random-init PATH` saves a fresh untrained net in the same format and exits
  (no Godot) — the T5 baseline.

**`tests/unit/test_warm_start.py` (new) — T1 (R29).** Seven tests, Godot-free
(mock games for the collection / end-to-end ones):

| Case | Test |
|---|---|
| cloning pi + v loss both drop on a fixed synthetic set | `test_cloning_loss_drops_on_fixed_set` |
| cloned policy mean lands near the recorded actions (MAE < 0.1) | `test_cloned_mean_is_near_the_recorded_action` |
| `log_std` reset to `INIT_LOG_STD` after cloning | `test_log_std_is_reset_after_cloning` |
| `log_std` left alone when `reset_log_std=False` (clone touches only pi/v) | `test_log_std_left_alone_when_not_reset` |
| `collect_demos` shapes + actions in [-1,1] + decoded power in [0.2,1] | `test_collect_demos_shapes_and_range` |
| warm checkpoint loads through the `--init` loader (`opponents.load_state_dict`) | `test_warm_checkpoint_loads_through_init_path` |
| full `warm_start()` over mock games, result loads via `--init` | `test_warm_start_end_to_end_over_mock_games` |

**`scripts/t5_warm_start.ps1` (new) — T5 (R53).** Records chaser demos over N real
games and clones -> `warm_start.pt`; saves a random-init net -> `random_init.pt`;
evaluates each vs the chaser over `-Episodes` games (>= 20) on one dedicated
headless Godot each; reports win rate and goal diff (mean return) for both and
passes only if the warm net wins more AND has a higher goal diff than random.
Each Godot is launched detached via WMI and only the pids it started are stopped.

## Results — all gates pass

- **`pytest -m "not live"`: 74 passed** (67 from Phase 3 + 7 new in
  `test_warm_start.py`).
- **`dotnet build`: Build succeeded, 0 Error(s)** (C# untouched; one pre-existing
  `AnimatedTexture` deprecation warning, not from this phase).
- **T4 live smoke** (`scripts/smoke_test_phase2.ps1 -Games 4`, 2 epochs x 1200
  total turns, timescale 32; 4 games because free RAM was ~4.5-5 GB, the plan's
  tight fallback):

  | metric | value |
  |---|---|
  | games | 4 |
  | total turns | 2400 in 1.92 min |
  | turns/min | 1251 |
  | goals (W+L) | 196 |
  | stuck turns | 0 |
  | restarts | 0 |
  | out of bounds | 6 (3% of games) |
  | python tracebacks | 0 |
  | power 0.2 vs 1.0 | 0.375 vs 7.185 (~19x) |

  **SMOKE PASS.**

- **T5 strength** (`scripts/t5_warm_start.ps1 -Games 4 -RecordTurns 6000
  -CloneEpochs 30 -Episodes 20`), warm-started net and random-init net each vs the
  chaser, BEFORE any PPO, 20 games each:

  | net | W / L / T | win rate | goal diff / game |
  |---|---|---|---|
  | warm-started | 16 / 4 / 0 | **80%** | **+0.610** |
  | random-init | 4 / 16 / 0 | 20% | -0.750 |

  **T5 PASS** — the warm start clearly beats the random net on both win rate and
  goal diff. Final cloning losses: pi_mse 0.089, v_mse 0.751.

## T5 first attempt failed — diagnosis and fix

The first T5 run FAILED: warm 30% / -0.54 vs random 35% / -0.47. It was recorded
with the chaser's DEFAULT settings, which are stochastic — per-shot aim `noise`
plus a randomised kickoff that moves only one (randomly chosen) player with wide
`kickoff_noise`. MSE cloning fits the MEAN of the demonstrations, so a noisy,
partly-multimodal target regresses to a blurred average (pi_mse plateaued at
~0.14), and `evaluate.py` plays the deterministic mean — a mushy chaser that was
no better than random noise at kickoff.

This is not a frame bug (the recording is confirmed learner-frame). The fix is to
clone a clean, repeatable target: the demo chaser now defaults to DETERMINISTIC
(`noise=0`, `kickoff_noise=0`), while the opponent side stays noisy so states
still vary. With the clean target, pi_mse converged lower (0.089) and the warm net
reproduces chaser play: 80% vs the chaser, well above the 20% random baseline.

## Scope note on task 4.4

Task 4.4 has two clauses. The first — "the warm-started net beats the random net
vs chaser before PPO" — is the brief's T5 gate and is met (80% vs 20%). The
second — "PPO from it is never worse than from scratch at the same turn count" —
needs a full paired PPO comparison (two multi-hour runs), which belongs to the
Phase 7 overnight run and its budget, not this phase's gates. It is deferred
there deliberately, not skipped silently.

## Free RAM check

Free RAM was ~4.5-5 GB throughout (advisory "tight"), so both T4 and T5 used the
4-game fallback the plan allows rather than 6. All Godot processes started by the
runs were stopped afterwards (verified 0 remaining); a quoting bug in the shared
dedicated-Godot stop matcher was fixed in `t5_warm_start.ps1` so it reliably kills
its own games.

## Notes for later phases

- `warm_start.py --random-init <path>` is a handy way to mint a fresh baseline
  checkpoint in the training format without a run.
- Start a warm PPO run with: `python training_loop.py --games 6 --launch
  --init-checkpoint checkpoints/warm_start.pt --opponent self ...`.
- The critic fit is weak (v_mse ~0.75) because returns here are sparse goal signal
  over a zero value baseline; PPO's own critic takes over from epoch 0, so this
  only needs to be a rough start, not a trained critic.
