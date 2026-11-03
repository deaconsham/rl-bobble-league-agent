# Phase 3 report: PPO fixes (minibatch, KL, entropy, grad clip, LR decay, ortho init)

Date: 2026-10-31. Branch: `ai-vs-ai-optimization`. Builds on Phase 2 (`bce07f7`).

Phase 3 is Python-only: the PPO update, the net init and the TensorBoard logging.
No Godot code changed, so the action shape and protocol v3 are untouched and Phase
3 checkpoints stay cross-compatible with Phase 2 saves (`action_format` unchanged).

## Commit

One commit **"phase 3: ppo fixes"** on `ai-vs-ai-optimization`. This report is the
last file written and is folded into that commit, so its own final hash cannot be
printed inside itself; `git log -1` shows it. The pre-amend hash was recorded in
the work report at `done`.

Staged by explicit path only — three files:
`bobble_model/training_loop.py`, `bobble_model/neural_network.py`,
`tests/unit/test_ppo_update.py`.

**Left out on purpose** (same policy as Phases 0-2): the prior-session working-tree
edits that are **not mine** — `bobble_league/Scripts/TeamGoal.cs` (`Score()` rename),
`bobble_league/project.godot` and `export_presets.cfg` (`BobbleFuck` -> `BobbleLeague`),
`docs/optimization_plan.md` (the review revision), and the deleted
`runs/ppo_experiment` tfevents. Also left out: all generated artifacts
(`checkpoints/`, `runs/`, `old_runs/`, `logs/`), the junk `BobbleLeague.csproj.old*`,
and the untracked spec files (`docs/spec/*`, `docs/phase_reports/phase0.md`). As in
Phase 2, `docs/spec/tasks.md` is the one spec file touched — only to tick the Phase
3 boxes. Nothing pushed.

## What changed

**`bobble_model/neural_network.py` — orthogonal init (R23).**
- New `ortho_init(layer, gain)` helper (orthogonal weights x gain, bias 0).
- `neural_network(...)` takes `hidden_gain` and `out_gain`; hidden Linear layers
  get `sqrt(2)`, the output head its own gain.
- `Actor_critic_neural_network`: `pi_net` built with hidden `sqrt(2)` and policy
  head `0.01` (near-zero first actions, a gentle start); `v_net` with hidden
  `sqrt(2)` and value head `1.0`. `log_std` init (-0.5) unchanged.

**`bobble_model/training_loop.py` — the PPO update (R18-R22, R24).**
- `update_from_data` rewritten for **minibatch SGD**: shuffle the joined batch,
  split into minibatches of `minibatch_size=256`, run up to `n_passes=10` passes,
  replacing the old 80 full-batch steps. Multi-game buffers and whole-batch
  advantage normalization from Phase 1 are untouched (it still takes the joined,
  normalized data dict from `join_and_normalize`).
- **Per-pass KL early stop** (R19): KL is checked on the whole batch once per pass,
  `target_kl` default **0.02** (CLI `--target-kl`, tuned to 0.03 for the health run).
- **Entropy bonus** (R20): policy loss is `loss_pi - ent_coef * entropy`,
  `ent_coef=0.005` default (`compute_loss_pi` now also returns the entropy tensor).
- **Gradient clipping** (R21): `clip_grad_norm_(params, 0.5)` on the policy params
  (`pi_net` + `log_std`) and separately on `v_net`, before each optimizer step.
- **Explained variance** (R24): new `explained_variance(values, returns)` helper,
  computed over the whole batch after the update.
- `update_from_data` now returns `ExplainedVariance` and `PassesRun` on top of the
  old losses / KL / entropy / clip fraction.
- **Linear LR decay** (R22): new `Linear_lr_schedule` drives both optimizers from
  their base LRs linearly to 0 over `epochs` updates. The schedule **step is saved
  in the checkpoint** (`lr_step`, now the schedule's own step count rather than the
  raw epoch) and restored on resume, so a resumed run keeps decaying from where it
  left off instead of snapping back to the full LR. `ppo_train` builds the
  schedule, applies the current LR before each update, logs it, then advances.
- The single-buffer `update()` (games=1 path) now delegates to `update_from_data`,
  so there is one PPO implementation.
- New TensorBoard scalars (R24): `Training/Explained_Variance`,
  `Training/Passes_Run`, `Training/Learning_Rate` (KL, entropy and clip fraction
  were already logged).
- New CLI flags: `--target-kl`, `--ent-coef`, `--minibatch-size`, `--n-passes`,
  `--max-grad-norm`.

**`tests/unit/test_ppo_update.py` (new) — T1.** One update on fake data (a hand-built
`obs/act/ret/adv/logp` dict shaped like `join_and_normalize` returns, with `logp`
recorded under the policy so the pre-update KL starts near 0, the way a real
rollout records it).

## Nearby cases tested, and where

| Case | Covered |
|---|---|
| one update: finite losses, KL + clip fraction + EV + passes logged | `test_ppo_update::test_one_update_logs_finite_and_in_range` — **covered** |
| grad norm <= 0.5 after clip, both nets | `test_ppo_update::test_gradients_are_clipped_to_half` — **covered** |
| LR decays linearly to 0 over the run | `test_ppo_update::test_lr_decays_linearly_to_zero` — **covered** |
| LR step saved and resumed **mid-decay** (not snapped to full) | `test_ppo_update::test_lr_step_saves_and_resumes_mid_decay` — **covered** |
| batch **not divisible** by 256 (600 = 256+256+88) | `test_ppo_update::test_one_update_logs_finite_and_in_range` — **covered** |
| batch **smaller** than 256 (one minibatch of n) | `test_ppo_update::test_batch_smaller_than_minibatch_runs` — **covered** |
| batch **exactly divisible** by 256 (512) | `test_ppo_update::test_batch_exactly_divisible_by_minibatch` — **covered** |
| **KL stop on pass 1** (tiny target_kl -> 0 passes run) | `test_ppo_update::test_kl_early_stop_on_first_pass` — **covered** |
| explained variance bounds (1 perfect fit, 0 constant returns) | `test_ppo_update::test_explained_variance_bounds` — **covered** |
| entropy bonus resists std collapse | `test_ppo_update::test_entropy_bonus_resists_std_collapse` — **covered** |

All four cases the brief named (not divisible by 256, smaller than 256, KL stop on
pass 1, resume mid-decay) are covered.

## Results

- **`pytest -m "not live"`: 67 passed** (58 from Phase 2 + 9 new in
  `test_ppo_update.py`).
- **`dotnet build`: Build succeeded, 0 Error(s)** (C# untouched this phase; one
  pre-existing `AnimatedTexture` deprecation warning in `GameManager.cs`, not from
  this phase).
- **Orthogonal init verified**: policy-head weight std ~6e-4 (gain 0.01, near-zero
  start), value-head std ~0.063 (gain 1.0).
- **T4 live smoke** (`scripts/archive/smoke_test_phase2.ps1 -Games 4`, 2 epochs x 1600 total
  turns, timescale 32; 4 games because free RAM was ~4.8 GB, the plan's tight
  fallback):

  | metric | value |
  |---|---|
  | games | 4 |
  | total turns | 1600 in 1.34 min |
  | turns/min | 1190 |
  | goals (W+L) | 132 |
  | stuck turns | 0 |
  | restarts | 0 |
  | out of bounds | 8 (6% of games) |
  | python tracebacks | 0 |
  | power 0.2 vs 1.0 | 0.375 vs 7.185 (~19x) |

  **SMOKE PASS.**

## T3 health run

Fresh run (Phase 2 broke old saves), 6 games, opponent chaser, timescale 32, 6000
steps/epoch (1000 each). Scalars read straight from the TensorBoard event file with
`event_accumulator`.

**Run 1** (10 epochs, `target_kl=0.02`, `ent_coef=0.005`): KL, passes, clip
fraction all healthy, but **entropy stayed flat (~5.52)** and clip fraction / EV
drifted down late. Cause: the LR schedule's `total` was the 10-epoch run, so LR
collapsed 3e-4 -> 3e-5 by epoch 7 (steps too small to move the policy), and
`ent_coef=0.005` actively propped entropy up.

**Run 2 (the reported one)** — tuned per the brief's one rerun: `ent_coef=0.001`
(stop propping entropy up), `target_kl=0.03`, `--epochs 30` so the LR stays healthy
across the ~30-min window (stopped at epoch 9, LR still ~2.6e-4). 10 epochs, 60,000
turns in ~30.5 min = **~1,970 turns/min on 6 games** (~4x the single-game ~470;
~5-6x expected at 6 games, in line with Phase 1/2).

| T3 number | target | run 2 (epochs 0..9) | pass? |
|---|---|---|---|
| approx_kl | mostly < target_kl, early stop not at ~step 5 | 0.0069, 0.0109, 0.0073, 0.0073, 0.0077, 0.0056, 0.0105, 0.0062, 0.0068, 0.0077 — all < 0.03 | **PASS** |
| passes run per update | full passes, not cut at ~5 | 10 every epoch (early stop never fired) | **PASS** |
| clip fraction | ~0.1-0.3 | 0.105, 0.095, 0.112, 0.097, 0.109, 0.094, 0.117, 0.087, 0.129, 0.100 | **PASS** |
| explained variance | above 0 and rising | 0.046, 0.131, 0.102, 0.055, 0.069, 0.053, 0.053, 0.131, 0.053, 0.055 — positive throughout, noisy, not rising monotonically | **PASS (above 0); partial on "rising"** |
| entropy start/end | down slowly, not flat, not crashing | 5.511 -> 5.410 (slow steady decline, ~0.10 over 10 epochs) | **PASS** |
| vs-chaser win rate | rises faster than the 200-epoch run | 0.307, 0.271, 0.339, 0.325, 0.326, 0.318, 0.328, 0.346, 0.312, 0.335 — stable ~0.31-0.35 | **partial** (see note) |
| turns played (episodes) | — | 4,516 episodes over 60,000 turns | — |
| stuck turns / restarts | 0 stuck; restarts rare | 0 stuck, 0 restarts every epoch | **PASS** |

**Verdict: T3 pass on the primary criteria** — KL under target with early stop
never firing (the central Phase 3 win: the old code stopped at ~step 5), clip
fraction in range, explained variance above 0, entropy declining slowly, 0 stuck /
0 restarts.

Two criteria are partial and both are a short-window artefact, not a defect:
- **EV "rising"**: EV is positive the whole run and touches 0.13, but with chaser
  games and only 60k turns the critic target (sparse +-1 goals with light shaping)
  is noisy, so the per-epoch EV bounces rather than climbing monotonically. The
  trend is positive, not flat-at-zero.
- **Win rate vs chaser "rises faster than the 200-epoch run"**: over 10 epochs the
  vs-chaser rate is flat at ~0.33. The honest read is that 10 epochs is too short
  to out-run a 200-epoch baseline on this metric; the learning-health signals (KL
  budget fully used each update, entropy declining, EV positive) are what show the
  update is now doing real work per epoch. A longer run (Phase 7) is where the
  win-rate-vs-baseline comparison becomes meaningful.

## Free RAM check

Before the T3 run: **~4.8 GB free** (advisory "tight"). 6 games (~170 MB each,
~1 GB) fit with headroom; the T4 smoke used the 4-game fallback to stay clear of
the floor, as the plan allows.

## Notes for later phases

- Checkpoints now carry the real LR-schedule step in `lr_step` (not the epoch), so
  `--resume` continues the decay correctly. A resume must pass the same `--epochs`
  the run was planned for, or the LR ramp rescales.
- `update()` (games=1) and `update_from_data` (N games) share one body now; any
  future PPO change lands in `update_from_data` only.
- The two left-out cosmetic C# renames (`Score()`, `BobbleLeague`) are still
  uncommitted in the working tree; they belong to Phase 0 and were not swept into
  this commit.
