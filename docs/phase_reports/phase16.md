# Phase 16: final clean-up and test

Branch `ai-vs-ai-optimization`. Closes the optimization plan: whole-repo clean-up,
docs brought to the final code, the full test gate, and the champion shipped as
the default model. Phases 13 and 15 were skipped on purpose (aim already fine;
the AI is no longer easy to beat). The human-vs-AI feel check needs a person and
is left waiting on the user (command below).

## 1. What was removed / moved, and why

The codebase was already well kept: every `training_loop.py` flag is wired into
`ppo_train`, the CLI is consistent across `training_loop.py`, `evaluate.py`,
`league_eval.py` and `launch_games.py` (shared `--godot` / `--project` /
`--shm-prefix` / `--step-timeout` / `--device` / `--use-layernorm` /
`--max-ep-len`), and there were no dead code paths or unused flags to delete. So
the clean-up was mostly tidying, not surgery.

- **Archived 8 one-off per-phase scripts** into `scripts/archive/`:
  `smoke_test_phase1/2/5/6/9.ps1`, `eval_epoch60.ps1`, `t5_warm_start.ps1`,
  `verify_no_oob.ps1`. Each is the source of one phase report's numbers, so they
  were kept for provenance (not deleted) and are not maintained;
  `scripts/archive/README.md` says so and maps each script to its report. The 18
  citations in the phase reports were repointed to the new paths (path-only
  edits). `scripts/` now holds only the two live tools, `smoke_test.ps1` and
  `proof_live_elo.ps1`. (Decision made with the user: keep provenance, clean the
  top level.)
- **Kept** `bobble_model/power_probe.py` (cited by `phase2.md`, a real standalone
  live check) and the Phase 12 feature flags `--input-norm`,
  `--random-start-prob`, `--reward-scale`, `--potential-shaping` (kept in code,
  default off: a T5 miss in Phase 12, so a plain launch reproduces Phase 11).
- **Left untouched**: all untracked run dirs (`p7_run`, `p10_run`, ...,
  `p14_run`), `bobble_model/p7_eval.py` (untracked run artifact), and every `.pt`
  file. `docs/optimization_plan.md` was also left alone — it already carried
  uncommitted working-tree edits from outside this phase, so this phase did not
  touch it.

## 2. CLI changes

- `evaluate.py` now defaults to the champion. New `default_model()` prefers
  `checkpoints/best_final.pt` and falls back to the newest `ppo_epoch_*.pt`; AI
  play and `--human` both use it when no `--checkpoint` is given. The module
  docstring and the `--checkpoint` help text were updated. No other CLI surface
  changed.

## 3. The champion

`bobble_model/p14_run/checkpoints/ppo_epoch_100.pt` (the Phase 14 long-run-2 best)
was **copied** (not moved) to `bobble_model/checkpoints/best_final.pt` and
verified to load: it carries LayerNorm, the value-norm stats and the input-norm
buffer, tagged action format 5. (`.pt` files are gitignored, so the champion is a
build artifact, not a committed file.)

## 4. Test results

| Check | Result |
|---|---|
| Full `pytest` (no `live` tests exist; this is the whole suite) | **205 passed** (203 baseline + 2 new `default_model` regression tests) |
| `dotnet build` | **0 errors** (1 pre-existing `AnimatedTexture` obsolete warning, out of scope) |
| Godot `--import` | ran (exit 0); the `.csproj` SDK bump it made was reverted BOM-less, back to `Godot.NET.Sdk/4.6.0`, git-clean |
| T4 live smoke, **6 games** | **PASS** — train 2 epochs × 6 games exit-clean, 0 restarts, turns settle mean ~3.8 s (p95 6.2 s) under the 10 s cap; eval ran; goals scored, 0 stuck turns, 0 out-of-bounds, 0 Python errors |

RAM: the box held ~5.3 GB free throughout, so all 6 games (and the 4 T5 games)
ran without dropping to 4. Only the Godots these runs started were stopped; 0
stray Godots after each run.

## 5. T5 strength (best_final.pt vs best_p11_ln.pt + the 4 anchors)

One within-run `league_eval` (so the anchored Elo scale is directly comparable —
it drifts ~20-50 pts run to run), 40 games per pairing, over 4 Godots. Anchors are
pinned: chaser 1000, defender 950, replay 850, random 800.

| Model | Elo | vs random | vs chaser | vs defender | vs replay |
|---|---|---|---|---|---|
| **best_final.pt** | **1104.0** | 95% (38-2-0) | 68% (26-11-3) | 77% (27-5-8) | 95% (36-0-4) |
| best_p11_ln.pt | 1031.2 | 92% (37-3-0) | 40% (16-13-11) | 48% (19-12-9) | 80% (32-4-4) |

(Win rate counts a draw as half a win; raw W-L-D in brackets.)

- **best_final.pt wins the run at 1104 Elo, +73 over best_p11_ln (1031)**, and beats
  it on every anchor pairing (notably chaser 68% vs 40% and defender 77% vs 48%).
- Head-to-head the two are within noise: best_final 19-21 best_p11_ln (so the
  0.55 best-gate reports "not best" on the h2h alone), which is why the 25-Elo
  anchor margin — not the noisy 20/40-game h2h — is the authoritative promotion
  gate (Phase 11/14 finding).
- **Repeated-move (replay) exploit:** against best_final the fixed replay move
  scored in 0 of 40 (36-0-4), i.e. it no longer works on the champion. The
  `replay_score_rate` the tool prints (20%) is for best_p11_ln (the "newest" of
  the two in the list), kept here for reference.

This matches the Phase 14 eval's conclusion (epoch 100 is the champion); the
absolute Elo differs from Phase 14's ~1156 by the usual run-to-run anchor drift,
but the ordering — best_final clearly ahead of best_p11_ln in the same run — is
what the gate judges, and it holds.

## 6. Human-vs-AI feel check — waiting on the user

This one needs a person (stop speed, the 10 s turn limit, kick power); it is not
faked here. To run it:

1. Open `bobble_league` in the Godot editor, open `Scenes/scene.tscn`, set the
   `GameManager`'s **CurrentPlayMode** to `HUMAN_VS_AI`, and run the scene. You are
   Team A: left-drag to aim, drag length sets power, right-click ends your turn.
2. In another terminal, from `bobble_model/`:

   ```
   python evaluate.py --human
   ```

   With no `--checkpoint` this plays `checkpoints/best_final.pt` with a shrunk
   sampling spread (std × 0.3) and no turn timeout. Human turns settle at
   `HumanStopSpeed` (0.3); a turn that never settles ends after `MaxTurnSeconds`
   (10 s). Check that stopping feels right, the 10 s limit never cuts a normal
   turn, and kick power feels proportional to drag length.

## 7. Commits (this phase, logically separate by path)

1. `chore: archive 8 one-off phase scripts under scripts/archive/` — the move, the
   archive README, and the 18 repointed citations.
2. `feat: default evaluate.py to the shipped champion best_final.pt` — `evaluate.py`
   + the two regression tests.
3. `docs: add README and bring docs/spec up to the final code` — `README.md` and
   the `docs/spec` addenda + stale-fact fixes.
4. `docs: Phase 16 report` — this file.

`docs/optimization_plan.md` was intentionally not committed (it carried unrelated
uncommitted edits; this phase leaves the plan alone). No `.pt` files committed
(gitignored). Nothing pushed.
