# Phase 6 report: Elo eval, held-out opponents, PFSP, reward-shaping fade

Date: 2026-10-31. Branch: `ai-vs-ai-optimization`. Builds on Phase 5 (`cb92abe`).

Phase 6 adds the tools to measure real strength and to pick opponents well:
league Elo, a fixed held-out test set, prioritized fictitious self-play (PFSP),
a fade of the per-turn shaping rewards, and play-time randomness. It is all
Python plus tests — no C# changes, so the wire protocol and checkpoints are
untouched.

## Commit

One commit **"phase 6: elo, pfsp and reward fade"** on `ai-vs-ai-optimization`.
This report is the last file written and is folded into that commit, so its own
final hash cannot be printed inside itself; `git log -1` shows it. The pre-commit
hash is recorded in the work report at `done`.

Staged by explicit path only — these files:
`bobble_model/elo.py` (new), `bobble_model/league_eval.py` (new),
`bobble_model/opponents.py`, `bobble_model/bobble_league_env.py`,
`bobble_model/training_loop.py`, `bobble_model/evaluate.py`,
`tests/unit/test_elo.py` (new), `tests/unit/test_opponents.py`,
`tests/unit/test_rewards.py`, `tests/mock/test_live_vs_live.py`,
`scripts/smoke_test_phase6.ps1` (new), `docs/spec/tasks.md` (Phase 6 boxes
ticked only), and this report.

**Left out on purpose** (same policy as Phases 0-5): the prior-session
working-tree edits that are **not mine** — `bobble_league/Scripts/TeamGoal.cs`,
`bobble_league/project.godot`, `export_presets.cfg`, `docs/optimization_plan.md`,
and the deleted `runs/ppo_experiment` tfevents. Also left out: all generated
artifacts (`checkpoints/`, `runs/`, `old_runs/`, `logs/`), the junk
`BobbleLeague.csproj.old*` backups, and the untracked spec/report files from
other phases (`docs/spec/design.md`, `requirements.md`, the other
`docs/phase_reports/*.md`). Nothing pushed.

## What changed

**`bobble_model/elo.py` (new) — Elo maths (R35).**
Pure maths, no Godot and no torch, so it is unit-testable on hand numbers.
`expected_score` (logistic, scale 400), `update_pair` (one game, `K*(actual -
expected)`, rating mass conserved), and `Elo_table` which folds `record_game` /
`record_match` results and pins **anchors** (the fixed held-out opponents) so the
scale does not drift between runs. `to_csv` writes `name,rating,anchor`, best
first.

**`bobble_model/league_eval.py` (new) — round-robin + strength report (R35,
R36, R37).**
Plays a round-robin of EXPLICIT checkpoint paths (never a glob, so an old or
deleted one is not swept in) plus the held-out anchors, over ONE dedicated Godot
game: "A vs B" is one env with the learner side = A and the opponent side = B,
exactly as `evaluate.py` pits the learner against one opponent. Only post-Phase-2
checkpoints are eligible — `load_state_dict` refuses an older `action_format`,
and an ineligible path is **skipped with its reason**, not crashed on. It reports
the newest checkpoint's win rate vs each held-out opponent and the replay
"winning move" score rate, applies the **best-checkpoint gate** (`beats_previous_best`,
R37: strictly >55% of the head-to-head, draws as half), and writes the Elo CSV.
A turn that never settles is counted a draw and the game reset (`--step-timeout`,
default 30 s), so one wedged game cannot block the whole league — the single-game
env has no crash-restart of its own, unlike `Vector_env`.

**`bobble_model/opponents.py` — PFSP, held-out opponents, play-time std (R36,
R38, R40).**
- `pfsp_weights(win_rates, power=2)`: the AlphaStar weighting `(1 - winrate)^2`
  normalized to a probability, with an eps floor so an all-beaten pool falls back
  to uniform instead of dividing by zero.
- `SelfPlayPool` gained a `pfsp` flag. `pick_snapshot_index()` draws the frozen
  copy by PFSP weight when on, uniform when off; `record_result(index, won)`
  folds a game outcome into that copy's win rate with an EMA; `last_index` is the
  copy drawn this episode. The ~15% chaser / ~30% live / ~55% frozen mix is
  unchanged — PFSP only changes WHICH frozen copy, not the slot split. `pfsp`
  and the win rates ride through `state()`/`load_state()` so a resume restores
  them.
- `DefenderOpponent` (parks in front of its own goal and clears the ball) and
  `ReplayOpponent` (replays a fixed recorded move, cycling) — both **held-out**.
  `TEST_ONLY_OPPONENTS = (random, chaser, defender, replay)` and
  `make_test_opponent` build them; the pool never draws from this set, only its
  own chaser / live / frozen slots, so a test opponent can never leak into
  training. The held-out chaser is the HARD chaser (no kickoff softening), so it
  is a real test distinct from the pool's training chaser.
- `PolicyOpponent` gained `std_scale`: with deterministic off it samples
  `mu + std_scale * std * noise`, so `evaluate.py --human` can play with a shrunk
  spread (R40).

**`bobble_model/bobble_league_env.py` — shaping fade + win-rate hook (R38, R39).**
- `RewardConfig.scaled(frac)` returns a copy with every per-turn SHAPING weight
  times `frac`, leaving `goal` untouched. `shaping_fade_fraction(epoch,
  total_epochs, fade_fraction=0.5)` is the multiplier: 1.0 at epoch 0, falling
  linearly to 0.0 at the halfway epoch, then 0 for the rest.
- `Vector_env._maybe_record_pool_result(i, scorer)` folds a frozen-copy game's
  outcome into that copy's PFSP win rate at game end, BEFORE the internal reset
  draws a new copy (the ordering that would otherwise lose `last_index`).

**`bobble_model/training_loop.py` — wiring (R38, R39).**
- `ppo_train` takes `pfsp`, `shaping_fade`, `shaping_fade_fraction`. The
  self-play pool is built with `pfsp=`. Each epoch, when `shaping_fade` is on, the
  env's reward config is set to `base.scaled(shaping_fade_fraction(epoch, epochs,
  frac))`, so the shaping fades over training while the goal reward stays.
- New CLI flags `--pfsp`, `--shaping-fade`, `--shaping-fade-fraction`.

**`bobble_model/evaluate.py` — play-time randomness (R40).**
`--human` builds the agent with `deterministic=False, std_scale=0.3` so one fixed
counter does not always work; AI_VS_AI eval keeps the deterministic mean unless
`--stochastic`.

## Tests

T1 + T2, all Godot-free, seeded where randomness matters.

**`tests/unit/test_elo.py` (new, T1 — R41):** 11 tests — expected-score is 0.5 at
equal ratings and sums to 1; the logistic matches hand math (a 400 lead is 10/11);
one update matches `K*(score - expected)` with rating mass conserved; a draw moves
the favourite down; `record_match` equals game-by-game; an anchor stays pinned but
still shapes the other side; more wins ranks higher; the CSV is ranked; the
constants are the standard 400 / 32; and the best gate needs strictly >55% (draws
as half, no games -> not best).

**`tests/unit/test_opponents.py` (extended, T1 — R34, R38):** 11 new tests — PFSP
weights favour the losers and handle the empty / all-beaten cases; `record_result`
moves the win rate by EMA and ignores a bad index; PFSP selection biases toward the
hard copy while uniform does not; the held-out set is the known fixed four and
**never appears in 3000 pool draws** (only chaser / live / self); `make_test_opponent`
builds each one and rejects an unknown; the defender and replay outputs are in range
and the replay repeats and wraps; the default replay is a full-power straight shot;
and `std_scale` shrinks the sampling spread while deterministic ignores it.

**`tests/unit/test_rewards.py` (extended, T1 — R39):** 6 new tests — `scaled` fades
every shaping weight but keeps the goal reward; `scaled(0)` turns all shaping off;
the fade fraction is 1.0 at the start and **0 at the halfway epoch** and stays 0
after (checked both for a non-integer and an integer midpoint); it is monotonic
non-increasing; and the degenerate runs (fade_fraction 0, one-epoch run) give 0.

**`tests/mock/test_live_vs_live.py` (extended, T2 — R38):** 1 new test — a
frozen-copy game through the real `Vector_env` records the outcome into the drawn
copy's PFSP win rate (proving the env -> pool recording path end to end against
mock Godot).

## Results — all gates pass

- **`pytest -m "not live"`: 118 passed** (89 from Phase 5 + 29 new: 11 elo, 11
  opponents, 6 rewards, 1 mock PFSP-via-env).
- **`dotnet build`: Build succeeded, 0 Error(s)** (one pre-existing
  `AnimatedTexture` deprecation warning in `GameManager.cs`, not from this phase).
  No C# changed this phase; the build gate is run because the plan requires it.

### T4 live smoke (`scripts/smoke_test_phase6.ps1 -Games 4`)

Train 4 epochs x 1200 total turns, self-play, **`--pfsp --shaping-fade`**,
timescale 32 (4 games because free RAM was ~4.9 GB, the plan's tight fallback),
then eval vs chaser on one dedicated game:

| metric | value |
|---|---|
| games | 4 |
| epochs | 4 (0..3) |
| episodes | 52 (W 21 / L 31) |
| restarts | 0 |
| stuck turns | 0 |
| out of bounds | 0 |
| LR ramp | 3.00e-4 -> 2.25e-4 -> 1.50e-4 -> 7.50e-5 (decaying) |
| Team B turns stored | 1200 / 890 / 717 / 360 per epoch |
| turn length (s) | mean ~3.3, p95 ~4.0-4.8, max ~7.2 |
| eval vs chaser | **W 9 / L 1, 90% win rate**, mean return 0.84 |

**T4 PASS.** 0 stuck turns, 0 restarts, 0 out-of-bounds, goals scored. The PFSP +
shaping-fade run completed all four epochs with no error against real Godot. The
Team B stored-turn count falls across epochs because the pool gains a snapshot
each epoch (snapshot-freq 1), so more games draw a frozen copy (no B-side storage)
rather than the live learner — the intended mix, same as Phase 5.

### T5 strength (`league_eval.py`, 8 games/pairing, max-ep-len 120)

Two fresh checkpoints (epoch 0, epoch 3) vs each other and the four held-out
anchors, on one dedicated Godot game. Elo written to `runs/<tag>/elo.csv`:

| player | Elo | anchor |
|---|---|---|
| chaser | 1000.0 | yes |
| ppo_epoch_0.pt | 951.2 | no |
| defender | 950.0 | yes |
| ppo_epoch_3.pt | 950.0 | no |
| replay | 850.0 | yes |
| random | 800.0 | yes |

Newest checkpoint (epoch 3) head-to-head:

| opponent | record | win rate |
|---|---|---|
| random | W6 L0 D2 | 75% |
| chaser | W6 L2 D0 | 75% |
| defender | W2 L3 D3 | 25% |
| replay | W0 L0 D8 | 0% (all draws) |
| epoch 0 (as A) | epoch 0 beat it W5 L0 D3 | — |

**T5 PASS.** A short Elo run produced sane ratings with the anchors in the
expected fixed order (chaser 1000 > defender 950 > replay 850 > random 800), which
is the Phase 6 T5 bar. epoch 3 beats the random and (harder, held-out) chaser
anchors 75% each.

**Reading the numbers honestly:** these are 4-epoch smoke checkpoints, not a
trained model, so the checkpoint Elo spread is small and the parked `defender` and
the never-scoring `replay` are hard for them — expected this early. The replay row
is all draws because `ReplayOpponent` fires the same full-power straight shot every
turn and never converts from kickoff, so both sides time out: the "replay score
rate" here is an artifact of the replay opponent, not a strength readout, and it is
the Phase 7 trained model that the replay-under-20% bar is meant to judge. The
mechanism is what Phase 6 delivers; the strength verdict is Phase 7's.

## Notes for Phase 7

- Train with `--pfsp --shaping-fade` and full saves every 10 epochs; run
  `league_eval.py` on the 20-epoch saves (explicit paths) to track Elo across the
  run.
- The `league_eval` single-game serial run is slow (~3 s/turn, one game); for a
  big league keep `--games` per pairing modest or script several dedicated games.
- The best-gate (`beats_previous_best`, >55% over 100) is ready to drive
  "promote to best" once the overnight checkpoints exist.
