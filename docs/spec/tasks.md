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

- [ ] 2.1 Decode each player as angle (`a * pi`, seam at "straight back toward own goal") and power (`0.2 + 0.8 * (p + 1) / 2`) in the team's mirrored frame; action stays 6 numbers. (R13)
- [ ] 2.2 `Player.Launch(powerScale)`: scale launch force instead of always normalizing; apply from `GameManager.ApplyTeamAction`. (R14)
- [ ] 2.3 `ChaserOpponent` and `evaluate.py --human` output angle + power; human drag length -> power. (R15)
- [ ] 2.4 Bump `PROTOCOL_VERSION` in `SharedMemory.cs` and `shared_memory_link.py` (still 120 bytes); old/new refuse each other. (R16)
- [ ] 2.5 **Test (T1, T4):** T1 angle/power decode + mirror-frame round trip; T4 live smoke; power 0.2 moves a player clearly less than power 1.0. (R17, R48, R49, R52)

---

## Phase 3: PPO fixes

- [ ] 3.1 Minibatch SGD in `update`: shuffle, 256-size minibatches, 10 passes; replace the 80 full-batch steps. (R18)
- [ ] 3.2 Keep KL early stop but per pass, `target_kl` 0.02-0.03. (R19)
- [ ] 3.3 Entropy bonus: `loss_pi - ent_coef * entropy`, `ent_coef = 0.005`. (R20)
- [ ] 3.4 `clip_grad_norm_(params, 0.5)` for both nets. (R21)
- [ ] 3.5 Linear LR decay to 0 over the run (schedule step saved for resume). (R22)
- [ ] 3.6 Orthogonal init in `neural_network.py`: sqrt(2) hidden, 0.01 policy head, 1.0 value head. (R23)
- [ ] 3.7 Log `approx_kl`, clip fraction and explained variance to TensorBoard. (R24)
- [ ] 3.8 **Test (T1, T3):** T1 one update on fake data (finite losses, KL + clip fraction in range, grads clipped, LR decays); T3 after ~30 min: KL early stop not at ~step 5, entropy falls steadily, vs-chaser win rate rises faster than the 200-epoch run. (R25, R48, R49, R51)

---

## Phase 4: warm start from the chaser

- [ ] 4.1 `warm_start.py`: chaser vs chaser on 6 games ~20 min, record `(obs, action)`. (R26)
- [ ] 4.2 Behaviour cloning (MSE on the mean) for the policy and critic fit on results; raise `log_std` back to its init value after cloning. (R27)
- [ ] 4.3 `training_loop.py --init warm_start.pt` starts from the warm-start checkpoint. (R28)
- [ ] 4.4 **Test (T1, T5):** T1 cloning loss drops on a fixed recorded set; T5 warm-started net beats the random net vs chaser before PPO, and PPO from it is never worse than from scratch at the same turn count. (R29, R48, R49, R53)

---

## Phase 5: more and better data per turn

- [ ] 5.1 Add a "live" slot to `SelfPlayPool`; mix ~55% frozen copies, ~30% live, ~15% chaser. (R30)
- [ ] 5.2 In live-vs-live games, store Team B's transitions (mirrored view), keeping per-team paths apart; never store a frozen copy's turns. (R31)
- [ ] 5.3 Raise `MaxPhysicsStepsPerFrame` and try `--timescale=48/64` only if one Godot does not max its core; physics unchanged. (R32)
- [ ] 5.4 Log turn length (game seconds) as a histogram; if most turns end well under 20 s, lower `MaxTurnSeconds` or add damping and re-check. (R33)
- [ ] 5.5 **Test (T1, T2, T4):** T1 both-sides buffer keeps teams apart + opponent mix matches weights over 10k draws; T2 and T4 pass; turn-length histogram in TensorBoard. (R34, R48, R49, R50, R52)

---

## Phase 6: measure strength, pick opponents well

- [ ] 6.1 `league_eval.py`: round-robin of checkpoints every 20 epochs, Elo per checkpoint, saved to CSV. (R35)
- [ ] 6.2 Held-out opponents never added to the pool: alternate chaser, "defender" bot, random bot, replay of the winning move. (R36)
- [ ] 6.3 Best gate: a new checkpoint must beat the previous best >55% over 100 games. (R37)
- [ ] 6.4 PFSP: weight old-copy selection by `(1 - winrate)^2`, keeping ~15% chaser and ~30% live. (R38)
- [ ] 6.5 Shaping fade: fade `RewardConfig` weights to ~0 over the first half of training. (R39)
- [ ] 6.6 Play-time randomness: sample with std x 0.3 in `evaluate.py --human`. (R40)
- [ ] 6.7 **Test (T1, T5):** T1 Elo math on known results + PFSP weights + shaping weight 0 at the halfway epoch; T5 runs on every 20-epoch save. (R41, R48, R49, R53)

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
