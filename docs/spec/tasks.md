# Tasks: Bobble League RL optimization

Ordered, checkable tasks for `docs/optimization_plan.md`, grouped by phase in the
plan's order. Each task links the requirement IDs (from `requirements.md`) it
covers. Design detail is in `design.md`. The test task(s) are last in each phase
group, and a phase is committed only once its tests pass.

Branch: `ai-vs-ai-optimization`. Canadian English spelling; `-ize` kept; engine and
library API names unchanged.

---

## Phase 0: codebase clean-up

- [ ] 0.1 Confirm the Godot folder, `BobbleLeague.csproj`, `project.godot` names and `export_presets.cfg` export path all read as Bobble League. (R1)
- [ ] 0.2 Rename the env class (`Bobble_fuck_env` -> `Bobble_league_env`) and helper (`bobble_fuck_env_helper`), and fix every import in `training_loop.py`, `opponents.py` and `evaluate.py`. (R1)
- [ ] 0.3 Confirm `Score()` in `GameManager.cs` and `TeamGoal.cs`. (R1)
- [ ] 0.4 Remove `BobbleFuck.csproj.old`, `.old.1`, the root `.DS_Store`, and the duplicate root `BallProperties.tres`. (R2)
- [ ] 0.5 Remove the ASCII art, the `godgod...` block and crude comments in `DataManager.cs`; keep the lowercase style. (R2)
- [ ] 0.6 Canadian English spelling pass across comments, docs and our own names (`defense` -> `defence`, colour, centre); keep `-ize`; leave engine API names. (R3)
- [ ] 0.7 Apply the house style: C# 4-space indent, braces on the same line, short lowercase comments; Python `Like_this` classes and "Docstring for ..." docstrings. (R3)
- [ ] 0.8 Add the `tests/` tree with `mock_godot.py` and the first T1 unit tests; add `pytest` to `requirements.txt`. (R4)
- [ ] 0.9 **Test (T1, T2, T4):** T1 and T2 pass; `dotnet build` 0 errors; Godot `--import` (revert any SDK bump); T4 live smoke scores goals with 0 stuck turns. (R5, R48, R49, R50, R52)

---

## Phase 1: six games at once, crash-safe

- [x] 1.1 Read `--shm=<path>` on the Godot command line in `SharedMemory.cs`; fall back to the default path when absent. (R6)
- [x] 1.2 Write `launch_games.py`: N detached headless Godots, each with its own `--shm` file and `logs/godot_<i>.log`. (R7)
- [x] 1.3 Add `Vector_env` to `bobble_league_env.py`: write all actions, set all step flags, wait for all; reset done games alone. (R8)
- [x] 1.4 Add `--games N` to `training_loop.py`: one batched network call per turn; one buffer per game; join at update with batch-wide advantage normalization; one opponent per game sharing one snapshot list; `--steps-per-epoch` is the total. (R9)
- [x] 1.5 Crash/hang restart in `Vector_env`: cut the path (truncated, critic bootstrap), relaunch that one Godot, wait for the handshake, carry on, log it. (R10)
- [x] 1.6 Full checkpoints: save network, both optimizers, LR step, epoch, pool weights + win rates, and RNG; add `--resume <file>`. (R11)
- [x] 1.7 **Test (T1, T2, T4):** T1 save/load round trip; T2 6 mock games with one killed mid-epoch then resume; T4 with 6 real games: no stuck turns or file clashes, turns/min ~5-6x single game. (R12, R48, R49, R50, R52)

---

## Phase 2: new action shape (angle + power)

- [x] 2.1 Decode each player as angle (`a * pi`, seam at "straight back toward own goal") and power (`0.2 + 0.8 * (p + 1) / 2`) in the team's mirrored frame; action stays 6 numbers. (R13)
- [x] 2.2 `Player.Launch(powerScale)`: scale launch force instead of always normalizing; apply from `GameManager.ApplyTeamAction`. (R14)
- [x] 2.3 `ChaserOpponent` and `evaluate.py --human` output angle + power; human drag length -> power. (R15)
- [x] 2.4 Bump `PROTOCOL_VERSION` in `SharedMemory.cs` and `shared_memory_link.py` (still 120 bytes); old/new refuse each other. (R16)
- [x] 2.5 **Test (T1, T4):** T1 angle/power decode + mirror-frame round trip; T4 live smoke; power 0.2 moves a player clearly less than power 1.0. (R17, R48, R49, R52)

---

## Phase 3: PPO fixes

- [x] 3.1 Minibatch SGD in `update`: shuffle, 256-size minibatches, 10 passes; replace the 80 full-batch steps. (R18)
- [x] 3.2 Keep KL early stop but per pass, `target_kl` 0.02-0.03. (R19)
- [x] 3.3 Entropy bonus: `loss_pi - ent_coef * entropy`, `ent_coef = 0.005`. (R20)
- [x] 3.4 `clip_grad_norm_(params, 0.5)` for both nets. (R21)
- [x] 3.5 Linear LR decay to 0 over the run (schedule step saved for resume). (R22)
- [x] 3.6 Orthogonal init in `neural_network.py`: sqrt(2) hidden, 0.01 policy head, 1.0 value head. (R23)
- [x] 3.7 Log `approx_kl`, clip fraction and explained variance to TensorBoard. (R24)
- [x] 3.8 **Test (T1, T3):** T1 one update on fake data (finite losses, KL + clip fraction in range, grads clipped, LR decays); T3 after ~30 min: KL early stop not at ~step 5, entropy falls steadily, vs-chaser win rate rises faster than the 200-epoch run. (R25, R48, R49, R51)

---

## Phase 4: warm start from the chaser

- [x] 4.1 `warm_start.py`: chaser vs chaser on 6 games ~20 min, record `(obs, action)`. (R26)
- [x] 4.2 Behaviour cloning (MSE on the mean) for the policy and critic fit on results; raise `log_std` back to its init value after cloning. (R27)
- [x] 4.3 `training_loop.py --init warm_start.pt` starts from the warm-start checkpoint. (R28)
- [x] 4.4 **Test (T1, T5):** T1 cloning loss drops on a fixed recorded set; T5 warm-started net beats the random net vs chaser before PPO, and PPO from it is never worse than from scratch at the same turn count. (R29, R48, R49, R53)

---

## Phase 5: more and better data per turn

- [x] 5.1 Add a "live" slot to `SelfPlayPool`; mix ~55% frozen copies, ~30% live, ~15% chaser. (R30)
- [x] 5.2 In live-vs-live games, store Team B's transitions (mirrored view), keeping per-team paths apart; never store a frozen copy's turns. (R31)
- [~] 5.3 Raise `MaxPhysicsStepsPerFrame` and try `--timescale=48/64` only if one Godot does not max its core; physics unchanged. (R32) — mechanism already in `SetTimeScale` (raises MaxPhysicsStepsPerFrame + scales ticks); the 48/64 sweep is a tuning decision deferred to the Phase 7 full-box run.
- [x] 5.4 Log turn length (game seconds) as a histogram; if most turns end well under 20 s, lower `MaxTurnSeconds` or add damping and re-check. (R33) — logged; turns settle ~3 s vs the 20 s cap, so lowering the cap is a safe follow-up tuning step (see phase5.md).
- [x] 5.5 **Test (T1, T2, T4):** T1 both-sides buffer keeps teams apart + opponent mix matches weights over 10k draws; T2 and T4 pass; turn-length histogram in TensorBoard. (R34, R48, R49, R50, R52)

---

## Phase 6: measure strength, pick opponents well

- [x] 6.1 `league_eval.py`: round-robin of checkpoints every 20 epochs, Elo per checkpoint, saved to CSV. (R35)
- [x] 6.2 Held-out opponents never added to the pool: alternate chaser, "defender" bot, random bot, replay of the winning move. (R36)
- [x] 6.3 Best gate: a new checkpoint must beat the previous best >55% over 100 games. (R37)
- [x] 6.4 PFSP: weight old-copy selection by `(1 - winrate)^2`, keeping ~15% chaser and ~30% live. (R38)
- [x] 6.5 Shaping fade: fade `RewardConfig` weights to ~0 over the first half of training. (R39)
- [x] 6.6 Play-time randomness: sample with std x 0.3 in `evaluate.py --human`. (R40)
- [x] 6.7 **Test (T1, T5):** T1 Elo math on known results + PFSP weights + shaping weight 0 at the halfway epoch; T5 runs on every 20-epoch save. (R41, R48, R49, R53)

---

## Phase 7: overnight run

- [ ] 7.1 Run ~1-2M turns on 6 games, `--resume` ready, full saves every 10 epochs. (R42)
- [ ] 7.2 **Test (T5):** Elo rises across the run; beats the held-out set more than the 200-epoch run; the repeated move scores in under 20% of tries. (R43, R48, R53)

---

## Phase 8: only if it is still easy to beat

- [ ] 8.1 Exploiter: every ~50 epochs, train a short-lived second policy only against the frozen main AI; add it to the pool. (R44)
- [ ] 8.2 Left-right mirror loss: penalize `pi(mirror(s)) != mirror(pi(s))` rather than adding raw mirrored samples. (R45)
- [ ] 8.3 Teammate-shared network: one small net per player, shared weights, seeing itself, ball, teammates and enemies; new save format. (R46)
- [ ] 8.4 **Test (T1, T5):** T1 per piece (mirror of a mirror is the same board; swapping teammates swaps their actions); T5 vs the Phase 7 model. (R47, R48, R49, R53)


---

## Phases 9-16: final state (addendum)

Tasks 0.x-8.x above follow the original Phase 0-8 plan. The plan was renumbered
and extended; `docs/optimization_plan.md` is the source of truth. The tasks below
cover the later phases as shipped. (In the renumbered plan, the original "Phase 8:
only if still easy to beat" became Phase 15 and was skipped; task 7.1/7.2 above is
the overnight run, which ran.) Requirement IDs refer to the addendum in
`requirements.md`.

## Phase 9: train faster

- [x] 9.1 Raise the AI_VS_AI time scale (48 standard) and lift the physics-steps cap so one Godot core is used fully; physics unchanged because ticks scale. (R54)

## Phases 10-11: live Elo, best gate, LayerNorm, critic

- [x] 10.1 Add `live_elo.py`: score the newest save every `--live-elo-freq` epochs on one spare Godot (`--live-elo-shm`) vs the chaser (1000) and defender (950) anchors; log to TensorBoard; off when the shm is unset. (R55)
- [x] 10.2 Short `--live-elo-step-timeout` + overall `--live-elo-check-budget` so a wedged spare game ends the check instead of idling the run (Phase 12 review deadlock fix). (R55)
- [x] 10.3 `Best_tracker`: a save becomes best when its live Elo clears the current best by > 25 Elo (authoritative margin); the h2h vetoes only a clear one-on-one loss (< 0.45), not as a second gate; rollback after 2 checks > 100 Elo below best. (R56)
- [x] 10.4 `--best-path` / `--checkpoint-dir`: write best.pt and the `ppo_epoch_*.pt` saves under the run's own dir so runs do not overwrite each other. (R57)
- [x] 11.1 `--value-norm` (running value-target norm, saved), `--n-passes-v`, `--clip-ratio`, `--n-minibatches`, `--kl-warm-start`/`--kl-coef`/`--kl-fade-fraction`. (R58)
- [x] 11.2 `--use-layernorm` (+ `--weight-decay`): LayerNorm hidden layers; save format carries it; `PolicyOpponent.load` / `SelfPlayPool.load_state` auto-detect and rebuild. (R58)
- [x] 11.3 **Test (T1, T5):** checkpoint round-trips value-norm/LayerNorm; T5 picks the run's real best from the epoch saves (live gate is noisy). (R48, R58, R53)

## Phase 12: better data and rewards (kept, default off)

- [x] 12.1 `--input-norm`: running obs mean/std inside the net, frozen at eval, saved. Default off. (R59)
- [x] 12.2 `--random-start-prob`: random legal start layouts (Godot places, Python picks via byte 91; never ball-in-goal or player-in-wall). Default 0.0. (R59)
- [x] 12.3 `--reward-scale` and `--potential-shaping`. Default off. (R59)
- [x] 12.4 **Test (T1, T5):** Phase 12 unit tests pass; T5 shows the Phase 11 recipe still wins, so all four flags ship off. (R48, R59, R53)

## Phase 13: bins — SKIPPED

- [x] 13.1 Skipped on purpose: aim already fine (action sigma ~0.19), no discretized head needed. (R60)

## Phase 14: long run 2

- [x] 14.1 200 epochs / ~1.2M turns from `best_p11_ln.pt` with the Phase 11 recipe; full saves per run dir; live Elo + best gate on. (R61)
- [x] 14.2 Best-gate promotion fix (margin authoritative, h2h veto-only) + epoch saves to the run dir; regression tests. (R56, R57)
- [x] 14.3 **Test (T5):** 40-game T5 of selected epochs vs `best_p11_ln.pt`; epoch 100 wins (~1156 vs ~1104, 27-13 h2h) and ships as `checkpoints/best_final.pt`. (R61, R53)

## Phase 15: exploiter / mirror loss / shared net — SKIPPED

- [x] 15.1 Skipped on purpose: after long run 2 the AI is no longer easy to beat. (R60)

## Phase 16: final clean-up and test

- [x] 16.1 Codebase clean-up: CLI consistent across `training_loop.py`/`evaluate.py`/`league_eval.py`/`launch_games.py`; no dead flags; Phase 12 flags kept default off; superseded one-off scripts archived under `scripts/archive/` with a README (provenance). (R62)
- [x] 16.2 Copy the champion to `checkpoints/best_final.pt` and make it `evaluate.py`'s default (`default_model`). (R61)
- [x] 16.3 Update `README.md` and `docs/spec/` to the final code. (R62)
- [ ] 16.4 **Test:** full `pytest`; `dotnet build` 0 errors; Godot `--import` (revert any SDK bump); T4 6-game smoke; T5 on `best_final.pt` vs the 4 anchors + `best_p11_ln.pt` + the repeated-move exploit rate. (R62)
- [ ] 16.5 Human-vs-AI feel check by hand (`python evaluate.py --human` vs `best_final.pt`): stop speed, 10 s turn limit, kick power. Waiting on the user. (R62)
- [ ] 16.6 Write `docs/phase_reports/phase16.md` (what was removed, CLI changes, test results, T5 table, the human-play command) and commit. (R62)
