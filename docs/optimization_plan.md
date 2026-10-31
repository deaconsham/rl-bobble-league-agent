# Bobble League RL: plan to play better and train faster

Date: 2026-10-31. Baseline: 200-epoch self-play run (`ppo_epoch_199.pt`), entropy 5.51 -> 5.40,
~470 turns/min on one Godot at 32x, AI easy to beat with one repeated move.

## Where we are (from the code)

| Area | Now | Problem |
|---|---|---|
| PPO update (`training_loop.py` `update`) | Full batch, 80 policy steps, stops at `1.5 * target_kl` (0.01) | Stops after ~5 steps. Very little learning per epoch. |
| Entropy bonus | None | Exploration is only the fixed starting noise. |
| Gradient clipping, LR schedule | None | Updates can spike; no late-run fine tuning. |
| Data per turn | Only the learner's side trains | The opponent side is wasted when it is the same network. |
| Env count | 1 Godot, ~1 of 12 CPU threads | CPU mostly idle. |
| Action | Direction only (`Player.Launch` normalizes) | AI cannot choose soft vs hard shots. |
| Eval | Mean action, vs chaser win rate | Deterministic, so one human trick wins every time. No strength rating. |

## Phase 1: PPO fixes (small code, biggest learning win)

Source: "The 37 Implementation Details of PPO", ICLR blog track 2022
(https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), and SB3 PPO defaults
(https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html).

1. Minibatch SGD: shuffle the batch, split into minibatches of 256, run 10 passes. Replace the 80 full-batch steps.
2. Keep KL early stop but per pass, with `target_kl = 0.02-0.03`.
3. Add entropy bonus to the policy loss: `loss = loss_pi - ent_coef * entropy`, `ent_coef = 0.005`.
4. Clip gradients: `clip_grad_norm_(params, 0.5)` for both nets.
5. Linear LR decay to 0 over the run.
6. Orthogonal init (gain sqrt(2) hidden, 0.01 policy head, 1.0 value head) in `neural_network.py`.
7. Log `approx_kl`, clip fraction, explained variance to TensorBoard to check each change.

Acceptance: KL early-stop no longer fires at step ~5; entropy goes down steadily; vs-chaser win rate
rises faster than the 200-epoch run at the same turn count.

## Phase 2: train faster

| Step | Gain | Work |
|---|---|---|
| 2a. 4 Godot processes, each with its own `BOBBLE_SHM_PATH` | ~4x turns/min | Python: a vector env that steps all 4, batches one network call. Launcher script starts 4 detached Godots. |
| 2b. Learner-vs-current-learner games train both sides | ~2x data in those games | Store the Team B transitions too (it already sees a mirrored view). Only for games vs the live network, not frozen copies (their actions are off-policy). |
| 2c. Raise `max_physics_steps_per_frame` and try `--timescale=48/64` | Up to ~2x | Only if one Godot does not max its CPU core. Physics stays the same since ticks scale. |
| 2d. Bigger batch: `steps_per_epoch = 4000` across the 4 envs | Better gradients, same wall time per epoch as now | CLI change. |

RAM: ~170 MB per extra Godot, ~2.8 GB free, so 4 fits and 6 is tight.

Evidence for 2a: Unity ML-Agents trained Snoopy Pop 7.5x faster with multiple instances
(https://unity.com/blog/engine-platform/training-your-agents-7-times-faster-with-ml-agents).
Alternative later: several arenas inside one Godot scene, as ML-Agents does with "training areas"
(https://unity-technologies.github.io/ml-agents/Learning-Environment-Design/). Saves RAM, but needs
scene and `DataManager` changes, so do it only if RAM runs out.

Expected: 4 envs x ~470 = ~1900 turns/min, ~2.7M turns/day (vs ~50k-200k in past runs).

Acceptance: `train.log` shows ~4x turns/min; no stuck turns or shm clashes across envs.

## Phase 3: play better

1. **Launch power in the action space.** Pass the action length through `Player.Launch` instead of
   normalizing, clamp to [0.2, 1]. Today every shot is full power.
2. **Left-right mirror augmentation.** The field is symmetric in X. Flip X of obs and actions to double
   data and push the policy toward symmetric play. For PPO, use a mirror loss (penalize
   `pi(mirror(s)) != mirror(pi(s))`) rather than raw extra samples, since raw mirrored samples are
   off-policy. Sources: Kaleidoscope Experience Replay (https://arxiv.org/pdf/1910.09959),
   mirror loss vs augmentation (https://arxiv.org/pdf/2403.04359).
3. **Smarter opponent picks (PFSP).** Pick old copies by how often the learner loses to them,
   e.g. weight `(1 - winrate)^2`, instead of uniform 80/20. From AlphaStar league training
   (https://www.researchgate.net/publication/336911787_Grandmaster_level_in_StarCraft_II_using_multi-agent_reinforcement_learning).
   Keep ~15% chaser.
4. **Exploiter.** Every ~50 epochs, train a short-lived second policy only against the frozen main
   AI. It finds tricks like your repeated move. Add it to the pool so the main AI learns to block
   them. Source: league exploiters and Minimax Exploiter (https://arxiv.org/html/2311.17190).
5. **Shaping decay.** Fade the per-turn shaping weights (`RewardConfig`) to ~0 over the first half
   of training so the AI optimizes goals, not "ball moved forward".
6. **Play-time randomness.** Sample with std x 0.3 in `evaluate.py --human` so one fixed counter
   does not always work. Quick, but does not make it smarter.

## Phase 4: measure strength

1. `league_eval.py`: round-robin of checkpoints every 20 epochs, Elo per checkpoint, saved to CSV.
2. Fixed test: replay your winning move as a scripted opponent; track how often it still scores.
3. Gate: new checkpoint must beat the previous best >55% over 100 games to become "best".

## Order and effort

| Order | Item | Effort |
|---|---|---|
| 1 | Phase 1 PPO fixes | ~1 h |
| 2 | 2a parallel envs + 2d batch size | ~2-3 h |
| 3 | 4.1 Elo eval | ~1 h |
| 4 | 2b both-sides data, 3.3 PFSP, 3.5 shaping decay | ~2 h |
| 5 | 3.1 launch power (needs Godot change + new training) | ~1 h |
| 6 | Overnight run, ~1-2M turns | ~8-12 h unattended |
| 7 | 3.4 exploiter, 3.2 mirror loss | ~3 h, only if the AI is still exploitable |

## Risks

- Launch power changes the action meaning, so old checkpoints will not transfer. Start fresh.
- More envs use more RAM; watch free memory with 4 running.
- Both-sides data can make self-play collapse into one style; keep the frozen pool and chaser in the mix.
