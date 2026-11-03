# Review report: Phases 8-12

Date: 2026-11-02. Branch: `ai-vs-ai-optimization`. Not pushed.

Independent review and full test of the Phases 8-12 code
(`d80b264..ca17755`, plus `e23cef0` and `c8c588f`), by a fresh reviewer. Scope:
the Python (`bobble_model/`) and C# (`bobble_league/Scripts/`) changes, the
MUST-FIX live-Elo decade-boundary deadlock, the `best.pt` live-gate noise, a full
test pass, `dotnet build`, and a 6-game T4 live smoke.

## Summary

| | |
|---|---|
| Real issues found | 3 (1 MUST-FIX deadlock, 1 latent mis-tag, 1 noise-prone gate) |
| Issues fixed, with tests | 3 |
| Bigger/risky issues left as a question | 0 (one documented limitation, see end) |
| `pytest` (full) | **195 passed** (was 189 at `ce4234c`; +6 net new/changed tests) |
| `dotnet build` | **0 errors** (1 pre-existing `AnimatedTexture` warning, not ours) |
| T4 live smoke, 6 games @ ts 48 | **PASS**: 2373 turns/min, 0 stuck, 0 out-of-bounds, 0 tracebacks |
| Live-Elo deadlock proof | **PASS**: a real run crossed 4 live-Elo boundaries with `--live-elo-shm`, exit 0, no hang |

## Commits (not pushed)

| Commit | What |
|---|---|
| `14ecb8f` | Pin `ACTION_FORMAT_VERSION` to the angle+power era (3), not the live protocol version; correct the test that asserted the mis-tag. |
| `e12b446` | **MUST-FIX**: stop the live-Elo decade-boundary deadlock — bounded timeout, spare-game relaunch, per-check budget, guarded resets; regression test + live proof script. |
| `66f1e0e` | Require an Elo margin (25) before `best.pt` is replaced, so live-gate noise cannot promote a weaker save. |

Each commit was staged by explicit path (the two that share `live_elo.py` /
`training_loop.py` were split hunk-by-hunk so no commit bundles an unrelated
concern), and the staged tree of each was confirmed to parse. Nothing pushed.
`docs/optimization_plan.md` (modified) and the untracked run folders / spec docs
were left untouched — they are not this item's work, same decision as the earlier
phase workers.

## Issue 1 (MUST-FIX): live-Elo decade-boundary deadlock

### Root cause

The in-run live-Elo eval (`training_loop.run_live_elo` -> `live_elo.score_vs_anchors`
-> `_play_match` -> `Bobble_league_env.step`) drives ONE long-lived spare Godot
game, created once at the start of the run and reused at every `--live-elo-freq`
boundary. Two facts combine into the hang:

1. **The spare env inherited the TRAINING `step_timeout` (120 s).** It was built
   with `step_timeout=step_timeout`, the same long value the training games use.
2. **The live-Elo path had NO hang-restart.** `Vector_env` relaunches a game that
   hangs or dies (`_restart_and_truncate`, `step_timeout` + `restart_tries`), but
   the live-Elo path had nothing equivalent: on a `TimeoutError` it scored the
   game a draw and moved on — to the SAME dead game.

So when the spare Godot genuinely wedged at a boundary (process alive but idle,
its log not growing — the Phase 12 symptom at epochs 60 and 70), every one of the
`live_elo_games` games against every anchor (and the head-to-head) waited out the
full 120 s before being scored a draw. With 20 games x 2 anchors (+ a possible
head-to-head) that is up to ~2 hours of the training process sitting in a polling
`time.sleep` loop — exactly "training process idle, spare-Elo game making no
progress, `train.log` stops advancing". It was never a true lock; it was an
unbounded, un-recovered wait.

Reproduced before the fix with the fake-Godot harness: a spare game wedged
mid-check made `score_vs_anchors` take `n_games * step_timeout` to return. The
protocol itself recovers after a hang clears (confirmed), but a genuinely wedged
Godot never clears, so a relaunch is required, not just a shorter wait.

### Fix (`e12b446`)

Make the live-Elo eval resilient the same way `Vector_env` is:

- **Short, dedicated timeout.** The spare env now uses `live_elo_step_timeout`
  (default **20 s**, CLI `--live-elo-step-timeout`), not the 120 s training value,
  so a wedge is noticed in seconds.
- **Relaunch the spare game.** `training_loop` builds a dedicated one-game
  `Game_launcher` for the spare game (only when `--launch`, i.e. we own Godot
  launching) and passes it as `live_elo_launcher`. On the first hang, `_play_match`
  calls a `relaunch()` closure that stops, relaunches and re-attaches a fresh env
  on the same shm, then the check carries on — mirroring `Vector_env`'s per-game
  restart. `Game_launcher` gained an explicit `shm_paths` / `log_tag` override so
  the spare game lives at the known `--live-elo-shm` path it can be relaunched on,
  and its logs stay out of the training games' `godot_<i>.log`.
- **Per-check budget.** Each check sets an overall wall-clock deadline
  (`live_elo_check_budget`, default **180 s**, CLI `--live-elo-check-budget`). Once
  spent, the remaining games are scored draws immediately, so a boundary can never
  idle the run — even with no launcher to relaunch (e.g. the spare game was brought
  up externally): the check bails out cleanly instead of hanging.
- **`max_hangs` bail-out + guarded resets.** After 2 hangs the game is treated as
  dead and the rest scored draws at once; every `env.reset()` in the eval is now
  timeout-guarded (`_reset_quietly`) so a reset-after-terminated timeout can no
  longer propagate out of the eval and crash the run (the epoch-60 native exit -1).

When the spare game is healthy, none of this fires — the normal path is unchanged
apart from the shorter timeout.

### Proof

- **Regression test** `tests/mock/test_live_elo_deadlock.py` (fake-Godot T2, no
  real engine), 3 cases: (a) a wedged game with no relaunch bails out in bounded
  time instead of `n_games * timeout`; (b) an already-spent budget short-circuits
  the whole check; (c) the relaunch path brings up a fresh healthy game and the
  eval recovers and scores real wins/losses (the game is relaunched exactly once).
- **Live run** `scripts/proof_live_elo.ps1`: a real 6-game training burst with the
  spare game ON (`--live-elo-shm`), `--live-elo-freq 2`, crossed **4 live-Elo
  boundaries** (epochs 0, 2, 4, 6), finished at epoch 6, **exit 0, 0 tracebacks,
  0 relaunches needed** (the spare game handshook cleanly every boundary — the very
  path that hung at epochs 60/70 in Phase 12). Wall clock ~11 min.

## Issue 2: `ACTION_FORMAT_VERSION` mis-tag (latent)

`phase12.md` records that `ACTION_FORMAT_VERSION` was "now pinned" after the v5
protocol bump, but the code still had
`ACTION_FORMAT_VERSION = Shared_memory_link.PROTOCOL_VERSION` (= 5). The loaders
already accept `action_format >= MIN_ACTION_FORMAT (3)`, so loading was NOT broken
— but the tag no longer meant "the action era this save belongs to": every save
written under v4/v5 was stamped 4/5 even though the action/obs contract had not
changed since v3, and a real future action change would not distinguish pre/post
saves. Fixed by pinning `ACTION_FORMAT_VERSION = 3` (`14ecb8f`), with a comment
explaining why it is decoupled from the live protocol. The test that asserted the
old mis-tag (`test_checkpoint.test_new_checkpoint_loads_through_both_paths`) was
corrected to assert the pinned value; `test_prior_protocol_checkpoint_still_loads`
already covered the `>= 3` load contract and is unchanged.

## Issue 3: `best.pt` live-gate noise

`phase11.md` found the 2-anchor / 20-game live-Elo gate carries ~20 Elo of
run-to-run noise and once ranked a weaker save (epoch 99, T5 921.5) over the
genuinely stronger epoch 80 (T5 1039.1), so the gate could hand a weaker `best.pt`
to the next phase. Chose the cheap, low-risk fix (a margin, not more games — more
games slows every check): `Best_tracker` now requires a challenger to clear the
current best by more than `BEST_MARGIN_ELO` (**25**, above the ~20 noise band) in
addition to winning the head-to-head, before it replaces `best.pt` (`66f1e0e`).
Tunable via `best_margin` (0 restores the old behaviour). Tests added in
`tests/unit/test_phase10.py`.

This reduces churn from noise but does not make the 2-anchor / 20-game gate
*accurate* — see the open limitation below.

## Code review of the Phase 8-12 changes (no further bugs found)

Reviewed `neural_network.py` (`Running_mean_std`, `Input_norm`, `RewardScaler`,
LayerNorm wiring), `opponents.py` (`PolicyOpponent.load` LayerNorm auto-detect,
`SelfPlayPool` resume), `bobble_league_env.py` (async `Vector_env.step`, wrap/clip,
legacy vs PBRS shaping), `training_loop.py` (value-norm denorm, resume,
save/load), and the C# `GameManager.cs` / `SharedMemory.cs` (v5 random-start / OOB
flags). Findings:

- **Checkpoint save/load gaps: none.** Normalizer stats (value-norm, reward-scale)
  save under their own top-level keys and load gracefully to the identity when a
  key is absent; input-norm buffers ride inside `model` and `load_model_state`
  tolerates a pre-Phase-12 save (asserts the only missing keys are `obs_norm.*`);
  LayerNorm is auto-detected from the state_dict on every load site; both
  optimizers and the RNG/pool restore on `--resume`. All covered by tests.
- **Phase-12-flags-default-off: confirmed.** `input_norm=False`,
  `random_start_prob=0.0`, `reward_scale=False`, `potential_shaping=False`, and
  `_shaping` dispatches to `legacy_shaping` when potential shaping is off, so a
  plain `python training_loop.py ...` reproduces the Phase 11 recipe. Pinned by
  `test_rewards.test_default_env_uses_legacy_shaping_not_pbrs`. The only
  `log_std` reset is the pre-existing `--init-checkpoint` path, which a plain
  launch never triggers.
- **Dead code / flag mistakes: none found** beyond Issue 2.

## Tests, build, smoke

- **`pytest` (full): 195 passed.** NOTE: the repo has **no `live`-marked tests** —
  `pytest -m "not live"` and a plain `pytest` collect the identical 195. The T3 /
  T4 / T5 "live" checks are the PowerShell scripts under `scripts/`, not pytest
  markers, so "full pytest including live tests" is the 195 unit+mock (T1+T2)
  tests. New/changed this review: +3 in `test_live_elo_deadlock.py`, +2 margin
  cases in `test_phase10.py`, 1 corrected assertion in `test_checkpoint.py`.
- **`dotnet build`: 0 errors**, 1 pre-existing `AnimatedTexture` obsolete warning
  (not ours; present since before this review). No C# changed in this item.
- **T4 live smoke, 6 games @ timescale 48** (`scripts/archive/smoke_test_phase9.ps1`):
  2373 turns/min, **0 stuck**, **0 out-of-bounds**, 0 tracebacks, 10 eval goals
  (W3/L7). Real `--launch` path, exercising the new `Game_launcher` change.

## Clean up

- Throwaway reproduction scripts were deleted; the only new files kept are the
  regression test (`tests/mock/test_live_elo_deadlock.py`) and the proof script
  (`scripts/proof_live_elo.ps1`).
- The Godot games launched by the smoke and proof runs were stopped (training
  games by `Vector_env.close()`, the spare live-Elo game by the new `finally` in
  `main()` / the proof script's launcher). No stray Godot or Python processes
  remain. My run-output folders (`logs/proof_live_elo`, `logs/p9_t4_ts48`,
  `runs/proof_live_elo`, `runs/p9_t4_ts48`, the temp shm dirs) were removed.

## Left open (documented, not a blocker)

1. **The live-Elo gate is still a 2-anchor / 20-game estimate.** The margin
   (Issue 3) stops noise from promoting a weaker save, but the gate's `best.pt`
   remains a noisy advisory signal; the authoritative champion is still the
   end-of-run 40-game, 4-anchor T5, which the conductor reads. A fuller fix
   (widen the live gate to 4 anchors / more games, or replace it with a short
   multi-anchor eval over the top few saves at run end) is a larger change left
   for a future phase, as `phase11.md` already suggested.
2. **Spare-game relaunch needs `--launch`.** When the spare live-Elo game is
   brought up externally (not via `--launch`), there is no launcher to relaunch a
   wedged game; the check then bails out on its budget rather than hang. This is
   safe (no deadlock either way) and the long overnight runs use `--launch`, so it
   is a convenience limitation, not a correctness one.
3. **Pre-existing, out of scope:** the 23 committed
   `runs/ppo_experiment/*.tfevents.*` binary logs and whether `runs/` should be
   gitignored (raised in `phase8.md`); `docs/optimization_plan.md` and the
   untracked spec docs. Untouched by this review.
