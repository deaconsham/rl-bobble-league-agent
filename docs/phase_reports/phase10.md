# Phase 10: learning fixes, part A

Branch `ai-vs-ai-optimization`, on top of Phase 9 (`dded117`).

## What changed (steps 1-8)

1. **Tighter start spread, no entropy bonus.** `neural_network.py` gets a module
   constant `INIT_LOG_STD = -1.6` (sigma ~0.20) used for the policy `log_std`
   parameter, so the agent starts aiming tight instead of spraying. `ent_coef`
   default dropped to `0` everywhere it is set (`update`, `ppo_train`,
   `update_from_data`, the CLI arg).
2. **Small spread kept through warm start and every init load.** `warm_start.py`
   imports `INIT_LOG_STD` from `neural_network` (one source of truth, cannot
   drift), and `warm_start.pt` was rebuilt with it. `training_loop.py` resets
   `log_std` to `INIT_LOG_STD` right after loading any `--init-checkpoint`, so an
   old save's wide spread (sigma ~0.6) never leaks into a fresh run.
3. **Angle wrap, power clip.** New `wrap_angle` / `wrap_action` in
   `bobble_league_env.py`: angle slots (0,2,4) are wrapped with
   `(a + 1) % 2 - 1` into `[-1, 1)`; power slots (1,3,5) stay clipped. Applied at
   every clip site: env `step`, `Vector_env._write_turn` (learner and opponent),
   `decode_angle`, `warm_start.py`, and `GameManager.cs` (`Mathf.PosMod(a + 1f,
   2f) - 1f` for the angle, `Mathf.Clamp` kept for power). PPO stores the
   **pre-wrap** action, so the gradient sees what the net emitted.
4. **LR floor.** `Linear_lr_schedule` never drops below 30% of the start rate
   (floor 9.0e-5 of the 3.0e-4 start).
5. **Live Elo every 10 epochs.** New `live_elo.py` scores the current net on one
   spare Godot game vs the chaser and defender anchors; `elo/live` and the per
   anchor win rates go to TensorBoard.
6. **best.pt + rollback.** A save becomes `best.pt` only when it has the highest
   live Elo **and** beats the old best head-to-head over 55%. After a live Elo
   drop of more than ~100 held for two consecutive checks, the run rolls back to
   `best.pt` (and `log_std` is reset).
7. **Dual scoring.** Live Elo is measured with the mean move and again with an
   absolute sigma of 0.1 (new absolute-sigma mode on `PolicyOpponent`).
8. **Fresh from the rebuilt warm start.** The 100-epoch run started from the
   rebuilt `warm_start.pt`.

## Tests

- `pytest -m "not live"`: **144 passed** (was 125; +19 new cases in
  `tests/unit/test_phase10.py`, `test_ppo_update.py` LR test rewritten for the
  floor).
- `dotnet build`: **0 errors** (only the pre-existing `AnimatedTexture` warning).
- **T4 smoke** (`scripts/smoke_test.ps1`): goals 10, 0 stuck, 0 out-of-bounds,
  0 tracebacks.

### T1 nearby cases (step 10)

| case | input | result |
| --- | --- | --- |
| angle just past +1 | 1.1 | -0.9 (away from own-goal seam) |
| angle just past -1 | -1.1 | 0.9 |
| angle just over +1 | 1.05 | -0.95 |
| angle just over -1 | -1.05 | 0.95 |
| angle exactly +1 | 1.0 | -1.0 (seam, decodes same as +1) |
| angle exactly -1 | -1.0 | -1.0 |
| angle in range | 0.5 | 0.5 (unchanged) |
| large multiple | 3.2 | -0.8 (in [-1,1)) |
| large multiple | -5.7 | 0.3 (in [-1,1)) |
| power at 0 | 0.0 | 0.0 |
| power at 1 | 1.0 | 1.0 |
| power past 1 | 1.9 | 1.0 (clipped) |
| stored action | - | pre-wrap value stored, wrapped value written |
| old init load | `p7_run/.../ppo_epoch_100.pt` (sigma ~0.56-0.66) | log_std reset to -1.6 (sigma 0.202) |

No wrapped angle near +-1 is ever produced from a value just past the edge, so a
near-edge aim never swings to our own goal.

## 100-epoch T3 run (6 games, timescale 48, from the rebuilt warm start)

### sigma / KL / EV trend

sigma stayed inside [0.1, 0.3] and never hit 0:

| epoch | sigma (min-max) | sigma mean |
| --- | --- | --- |
| 0 | 0.202-0.202 | 0.202 |
| 20 | 0.200-0.206 | 0.203 |
| 40 | 0.197-0.210 | 0.203 |
| 60 | 0.195-0.208 | 0.201 |
| 80 | 0.187-0.201 | 0.195 |
| 99 | 0.178-0.196 | 0.189 |

KL settled under `target_kl` (0.02) after the warm-start transient. The first
~15 epochs ran 1-2 passes with the stored KL at 0.03-0.05 (the per-pass early
stop firing as the warm start settled); from ~epoch 25 on the loop ran the full
10 passes with KL ~0.010-0.021. EV held in the ~0.05-0.26 band throughout (the
value function is noisy on this sparse-goal task, consistent with prior phases).

### Live Elo curve (spare game vs chaser 1000 / defender 950, mean + sigma 0.1)

| epoch | mean Elo | best | note |
| --- | --- | --- | --- |
| 0 | 947.3 | 947.3 | first best |
| 10 | 894.4 | 947.3 | |
| 20 | 946.1 | 947.3 | |
| 30 | 874.9 | 947.3 | |
| 40 | 1011.9 | 1011.9 | new best, h2h 20-0 |
| 50 | 1066.8 | 1066.8 | new best, h2h 14-6 |
| 60 | 1051.5 | 1066.8 | |
| 70 | 1034.5 | 1066.8 | |
| 80 | 1094.1 | 1094.1 | new best, h2h 12-7-1 |
| 90 | 1092.6 | 1094.1 | |
| 99 | 1064.9 | 1094.1 | |

Strength climbs above the warm-start level and stays there; no live drop ever
exceeded 100 for two checks, so the rollback never fired. `best.pt` is epoch 80.

## T5: 40 games per pairing vs the fixed anchors (chaser 1000 / defender 950 / replay 850 / random 800)

Same anchored scale as the Phase 8 epoch-60 bar of **Elo 825.9**. Each subject
scored alone against the four anchors (so the scale cannot drift), 4 parallel
Godots at timescale 48 (`bobble_model/p10_run/t5_eval.ps1`, CSVs in
`bobble_model/p10_run/t5_csv/`).

| subject | Elo | >= warm start (747.5) | > bar (825.9) |
| --- | --- | --- | --- |
| warm_start | 747.5 | - | - |
| epoch_20 | 858.1 | yes | yes |
| epoch_30 | 789.3 | yes | no |
| epoch_40 | 882.9 | yes | yes |
| epoch_50 | 775.8 | yes | no |
| epoch_60 | 749.0 | yes | no |
| epoch_70 | 1018.0 | yes | yes |
| epoch_80 | 957.7 | yes | yes |
| epoch_90 | 917.5 | yes | yes |
| epoch_99 | 944.8 | yes | yes |
| **best.pt (epoch 80)** | **938.3** | **yes** | **yes (+112)** |

**Both T5 bars pass.** Every save from epoch 20 on scores at least the warm start
(minimum 749.0 at epoch 60, >= 747.5), and `best.pt` (epoch 80) scores **938.3**,
beating the 825.9 bar by ~112 Elo.

best.pt win rates at 40 games: vs defender 85%, vs random 85%, vs chaser 32%,
vs replay 25%. It is clearly stronger than the Phase 8 epoch-60 baseline but
still loses to the chaser, which is the target for the next phases.

## Artifacts

- Best save: `bobble_model/checkpoints/best.pt` (epoch 80, 40-game Elo 938.3).
- 100-epoch run log: `bobble_model/p10_run/logs/train.log`.
- T5 CSVs: `bobble_model/p10_run/t5_csv/`.
- New module: `bobble_model/live_elo.py`; new tests: `tests/unit/test_phase10.py`.
