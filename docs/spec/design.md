# Design: Bobble League RL optimization

Companion to `requirements.md`. Describes the architecture, the data flow, the
per-phase design, the file-by-file changes and the risks for turning
`docs/optimization_plan.md` into working code. Names, offsets and signatures below
were read from the real code in `bobble_model/` and `bobble_league/` as of
2026-10-31.

Spelling is Canadian English; `-ize` endings are kept; engine and library API
names are left as they are.

---

## 1. Architecture and data flow

The system is one Godot physics game per environment, bridged to one Python PPO
trainer over a file-backed shared-memory buffer.

```
+-----------------+     120-byte mmap (protocol v5)      +----------------------+
|  Godot (C#/.NET)|  <-- obs, reward, done, scorer -----  |  Python (PyTorch)    |
|  bobble_league/ |  --- action, opp action, step/reset ->|  bobble_model/       |
|                 |                                        |                      |
|  GameManager    |                                        |  Vector_env (N games)|
|  DataManager    |                                        |   -> PPO_buffer x N  |
|  SharedMemory   |                                        |   -> Actor_critic NN |
|  Player / Ball  |                                        |   -> update() (PPO)  |
+-----------------+                                        +----------------------+
```

### 1.1 The 120-byte shared-memory protocol (v5)

Defined twice and kept in lockstep: `bobble_league/Scripts/SharedMemory.cs`
(`PROTOCOL_VERSION = 5`) and `bobble_model/shared_memory_link.py`
(`Shared_memory_link`). The file is a plain mmap; Godot creates and zeroes it on
boot, then writes the version byte. Python refuses to attach to an older version.
The version has risen over the phases: v2 was the original launch-vector action,
v3 moved to angle+power (section 4.2), v4 added the per-turn length telemetry
(byte 120-123), and v5 added the random-start and out-of-bounds flags. A
checkpoint tagged action format >= 3 (angle+power) still loads; later bumps only
add side-channel fields, so they do not break old saves.

| Bytes | Field | Size | Direction |
|---|---|---|---|
| 0-55 | observation | 14 x f32 | Godot -> Python |
| 56-79 | learner action | 6 x f32 | Python -> Godot |
| 80-83 | reward | f32 | Godot -> Python |
| 84 | done | u8 (1 = goal) | Godot -> Python |
| 85 | step request | u8 | Python sets, Godot clears |
| 86 | reset request | u8 | Python sets, Godot clears |
| 87 | learner team | u8 (1 or 2) | Godot -> Python |
| 88 | scorer | u8 (0 none, 1 learner, 2 opponent) | Godot -> Python |
| 89 | play mode | u8 (AI_VS_AI / HUMAN_VS_AI / HUMAN_VS_HUMAN) | Godot -> Python |
| 90 | opponent action valid | u8 | Python -> Godot |
| 91 | pad | 1 | -- |
| 92-115 | opponent action | 6 x f32, opponent's own frame | Python -> Godot |
| 116 | protocol version | u8 | Godot -> Python |
| 117-119 | pad | 3 | -- |

Handshake per turn: Python writes the action (and opponent action, with byte 90 =
1), sets the step flag (byte 85), then polls until Godot clears it. Reset uses byte
86 the same way. A memory barrier on the C# side guards against the flag dropping
before the payload is written.

### 1.2 Observation and frames

`DataManager.cs` builds a 14-float observation in the learner's frame:
`[ball_x, ball_z, f1_x, f1_z, f2_x, f2_z, f3_x, f3_z, e1_x, e1_z, e2_x, e2_z, e3_x, e3_z]`.
Every team sees itself attacking -Z: `DataManager.FrameZ(team)` returns +1 for team
1 and -1 for team 2, and the Z coordinate is multiplied by that sign. Team 2's view
is therefore the Z-mirror of team 1's, so one policy plays either side.
`opponent_view(raw_obs)` in the env mirrors Z and swaps friendly/enemy.

Python normalizes obs to roughly [-1, 1] with
`OBS_SCALE = [FIELD_HALF_X, FIELD_HALF_Z] * 7` (`FIELD_HALF_X = 10`,
`FIELD_HALF_Z = 18.5`, `GOAL_Z = 16.5`).

### 1.3 Action (today)

6 numbers, `(x, z)` launch direction per player. `Player.Launch()` normalizes the
input, so only direction matters and the magnitude of the three vectors is thrown
away. `GameManager.ApplyTeamAction` applies `FrameZ` to the Z component. Phase 2
replaces this with angle + power (section 4.2).

### 1.4 PPO trainer

`bobble_model/training_loop.py` holds `PPO_buffer` (GAE-Lambda, gamma 0.99, lam
0.97, `discount_cumsum`), `compute_loss_pi` (clipped surrogate, clip 0.2),
`compute_loss_v` (MSE), `update` (today: 80 full-batch policy steps, early stop at
`1.5 * target_kl` with `target_kl = 0.01`, then 80 value steps) and `ppo_train`
(the epoch loop). `neural_network.py` holds `Actor_critic_neural_network`: `pi_net`
and `v_net` are `[obs_dim] + [256, 256] + [out]` MLPs with Tanh, and `log_std` is a
parameter initialized to -0.5. Two Adam optimizers: `pi_optimizer` over
`pi_net.parameters() + [log_std]`, `vf_optimizer` over `v_net.parameters()`.

### 1.5 Opponents

`bobble_model/opponents.py`: `StillOpponent`, `RandomOpponent`, `ChaserOpponent`
(scripted, softened at kickoff), `PolicyOpponent` (frozen net), `SelfPlayPool`
(keeps the last `max_size` snapshots, picks the chaser with `scripted_prob`, else a
random snapshot). `make_opponent(spec)` resolves the `--opponent` string.

---

## 2. Current state and gaps (from the code)

| Area | Now | Gap |
|---|---|---|
| PPO `update` | Full batch, 80 policy steps, early stop at `1.5 * target_kl (0.01)` | Stops after ~5 steps; little learning per epoch. |
| Entropy bonus | None | Exploration is only the fixed starting noise. |
| Grad clip, LR schedule | None | Updates can spike; no late-run fine tuning. |
| Init | PyTorch default | No orthogonal init. |
| Data per turn | Only the learner's side trains | Opponent side wasted when it is the same net. |
| Opponents | 80% frozen copy, 20% chaser | No live-vs-live, so both-sides data has no source. |
| Env count | 1 Godot, ~1 of 12 threads | CPU idle. |
| Action | 6 numbers, normalized | Magnitude thrown away; no soft vs hard shots. |
| Saves | Weights only | No optimizer / pool / RNG, so no resume. |
| Crashes | One Godot dies -> run dies | Fragile overnight. |
| Eval | Mean action, win rate vs chaser | Deterministic; no strength rating. |
| Tests | None in the repo | Checked by hand. |

---

## 3. Phase 0 design: clean-up (covers R1-R5)

Rename-only and comment-only edits, done first so every later change builds on the
final names.

File-by-file:
- `bobble_fuck/` -> `bobble_league/` (done, staged); `BobbleFuck.csproj` ->
  `BobbleLeague.csproj`, with `assembly_name`/`config/name` in `project.godot` and
  the export path in `export_presets.cfg`.
- `bobble_model/bobble_league_env.py`: the file is renamed, but the class is still
  `Bobble_fuck_env` and the helper `bobble_fuck_env_helper`; rename the class (e.g.
  `Bobble_league_env`) and helper, and update `training_loop.py`
  (`from bobble_fuck_env import bobble_fuck_env_helper`), `opponents.py`
  (`from bobble_fuck_env import normalize_obs, split_obs`) and `evaluate.py`
  (`from bobble_fuck_env import Bobble_fuck_env`).
- `GameManager.cs` / `TeamGoal.cs`: `Score()` is already in place; keep it.
- `DataManager.cs`: remove the ASCII-art block and the `godgod...` string and crude
  comments; keep the lowercase style.
- Spelling pass for Canadian English across comments, docs and our own names
  (`defense` -> `defence` in `RewardConfig` and the env shaping comment, colour,
  centre); keep `-ize`; leave engine API names.
- Add `tests/` with `mock_godot.py` and the first T1 unit tests; add `pytest` to
  `requirements.txt`.

Risk: the class/helper rename touches three Python modules; a missed import breaks
the whole stack, so the T2 import smoke is the guard.

---

## 4. Phase design

### 4.1 Phase 1: six games, crash-safe (covers R6-R12)

- `SharedMemory.cs`: add a `--shm=<path>` command-line read in the Godot args
  (same place `GameManager.ReadCommandLine` parses `--mode` / `--timescale`), and
  pass the resolved path into `SharedMemory` instead of only `ResolvePath()`. This
  also fixes the temp-folder mismatch, because Python passes the full path.
- `launch_games.py` (new): start N detached headless Godots, each with
  `--shm=<temp>/bobble_league_mmap_<i>.dat` and logging to `logs/godot_<i>.log`.
  On Windows, launch detached via `Invoke-CimMethod Win32_Process Create` so the
  process is not a child of the shell.
- `bobble_league_env.py`: add `Vector_env` holding N `Shared_memory_link`s. `step`
  writes all actions, sets all step flags, waits for all; `reset` resets per game.
  Done games reset independently.
- `training_loop.py`: `--games N`. One `ac.step` call batched over N obs per turn.
  N `PPO_buffer`s (one per game) so GAE paths stay separate; joined at update, with
  advantages normalized over the whole batch. `--steps-per-epoch` is the total
  across games (6000 for 6 games).
- Crash restart in `Vector_env`: on process exit or step timeout, `finish_path`
  with the critic bootstrap (truncated), relaunch that one Godot, wait for the
  handshake, continue the others. Log each restart.
- Full checkpoints: change `torch.save` from a bare `state_dict()` to a dict with
  `model`, `pi_optimizer`, `vf_optimizer`, `lr_step`, `epoch`, `pool` (snapshot
  weights + win rates), and `rng` (Python/NumPy/PyTorch). `--resume <file>` loads
  all of it. `load_state_dict` in `opponents.py` already unwraps `{'model': ...}`,
  so keep that shape backward-compatible.

### 4.2 Phase 2: angle + power (covers R13-R17)

- Decode in the env (or a shared helper): for each player, `angle = a * pi` in the
  team's mirrored frame with the -1/+1 seam at "straight back toward own goal";
  `power = 0.2 + 0.8 * (p + 1) / 2`. Action stays 6 numbers.
- `Player.Launch(float powerScale)`: scale `launchForce` by `powerScale` instead of
  always normalizing to full. `GameManager.ApplyTeamAction` converts angle+power to
  the input vector (still applying `FrameZ` to Z).
- `ChaserOpponent.act` and `evaluate.py --human`: emit angle + power; for humans,
  drag length -> power.
- `SharedMemory.cs` and `shared_memory_link.py`: bump `PROTOCOL_VERSION` to 3 (still
  120 bytes) so old and new refuse each other on byte 116.

Risk: human play changes (drag length now sets power). Old checkpoints stop
transferring, so training starts fresh after this phase.

### 4.3 Phase 3: PPO fixes (covers R18-R25)

In `training_loop.py` `update`:
- Shuffle the joined batch, split into minibatches of 256, run 10 passes (replace
  the 80 full-batch steps). Keep the KL early stop but per pass, `target_kl`
  0.02-0.03.
- Entropy bonus: `loss = loss_pi - ent_coef * entropy`, `ent_coef = 0.005`.
- `torch.nn.utils.clip_grad_norm_(params, 0.5)` for both nets before each step.
- Linear LR decay to 0 over the run (a `LambdaLR` or manual schedule; its step is
  saved for resume).
- Log `approx_kl`, clip fraction, explained variance to TensorBoard (clip fraction
  and KL already computed in `compute_loss_pi`; add explained variance from
  returns/values).

In `neural_network.py`: orthogonal init, gain sqrt(2) on hidden layers, 0.01 on the
policy head, 1.0 on the value head.

### 4.4 Phase 4: warm start (covers R26-R29)

- `warm_start.py` (new): chaser vs chaser on the 6 games for ~20 min, record
  `(obs, action)`.
- Behaviour cloning: MSE of the policy mean against recorded actions; fit the critic
  on game results. After cloning, raise `log_std` back to its init value (-0.5) so
  PPO still explores.
- `training_loop.py --init warm_start.pt`: the existing `init_checkpoint` path loads
  weights; reuse it.

Risk: warm start can lock in the chaser's style; raising `log_std` after cloning is
the guard.

### 4.5 Phase 5: more and better data (covers R30-R34)

- `SelfPlayPool`: add a "live" slot that plays the current learner. Mix ~55% frozen
  copies, ~30% live, ~15% chaser (extend `reset()` selection).
- In live-vs-live games, store Team B's transitions too, using its mirrored view
  (`opponent_view`). Never store a frozen copy's turns (off-policy). Keep per-team
  paths separate in the buffer.
- Raise `Engine.MaxPhysicsStepsPerFrame` and try `--timescale=48/64` only if one
  Godot does not max its core. Physics is unchanged because ticks scale with
  timescale (`GameManager.SetTimeScale`).
- Log turn length in game seconds as a TensorBoard histogram; if most turns end well
  under `MaxTurnSeconds` (20 s), lower the cap or add damping and re-check.

Risk: both-sides data can collapse self-play into one style; keeping the frozen
pool and chaser in the mix guards against it. More games use more RAM; watch free
memory with 6 running.

### 4.6 Phase 6: strength and opponent choice (covers R35-R41)

- `league_eval.py` (new): round-robin of checkpoints every 20 epochs, Elo per
  checkpoint, saved to CSV.
- Held-out opponents never added to the pool: a chaser with different settings, a
  "defender" bot in front of its goal, a random bot, and a replay of the winning
  move.
- Best gate: a new checkpoint must beat the previous best >55% over 100 games.
- PFSP: weight old-copy selection by `(1 - winrate)^2` instead of uniform, keeping
  ~15% chaser and ~30% live.
- Shaping fade: fade `RewardConfig` weights to ~0 over the first half of training
  (a schedule hit by `test_rewards.py`).
- Play-time randomness: sample with std x 0.3 in `evaluate.py --human`.

### 4.7 Phase 7: overnight run (covers R42-R43)

~1-2M turns on 6 games, `--resume` ready, full saves every 10 epochs. Validated by
T5 at the end.

### 4.8 Phase 8: only if still easy to beat (covers R44-R47)

- Exploiter: every ~50 epochs, a short-lived second policy trained only against the
  frozen main AI, then added to the pool.
- Left-right mirror loss: penalize `pi(mirror(s)) != mirror(pi(s))` (X symmetry)
  rather than adding raw mirrored samples.
- Teammate-shared network: one small net per player with shared weights, seeing
  itself, the ball, teammates and enemies; changes the save format again.

---

## 5. File-by-file change map

| File | Phase(s) | Change |
|---|---|---|
| `bobble_league/Scripts/SharedMemory.cs` | 1, 2 | `--shm` path; bump `PROTOCOL_VERSION` |
| `bobble_league/Scripts/GameManager.cs` | 2, 5 | angle+power apply; `MaxPhysicsStepsPerFrame`; `MaxTurnSeconds`/damping |
| `bobble_league/Scripts/DataManager.cs` | 0 | remove ASCII art / crude comments |
| `bobble_league/Scripts/Player.cs` | 2 | `Launch(powerScale)` |
| `bobble_league/Scripts/TeamGoal.cs` | 0 | keep `Score()` |
| `bobble_model/shared_memory_link.py` | 2 | bump version; keep offsets |
| `bobble_model/bobble_league_env.py` | 0, 1, 2, 5 | rename class/helper; `Vector_env`; angle+power decode; both-sides store |
| `bobble_model/neural_network.py` | 3, 8 | orthogonal init; shared-teammate net |
| `bobble_model/training_loop.py` | 0-6 | import fix; `--games`; minibatch/entropy/clip/LR; `--init`; full saves/resume; shaping fade |
| `bobble_model/opponents.py` | 0, 2, 5, 6, 8 | import fix; chaser angle+power; live slot; PFSP; exploiter |
| `bobble_model/evaluate.py` | 0, 2, 6 | import fix; angle+power human; std x 0.3 |
| `bobble_model/launch_games.py` | 1 | new: N detached headless Godots |
| `bobble_model/warm_start.py` | 4 | new: record + behaviour cloning |
| `bobble_model/league_eval.py` | 6 | new: Elo round-robin + held-out set |
| `bobble_model/requirements.txt` | 0 | add `pytest` |
| `tests/` | 0, 1, 3, 5, 6, 8 | T1 unit + T2 mock (`mock_godot.py`) |
| `scripts/smoke_test.ps1` | 0 | new: T4 live smoke |

---

## 6. Risks (collected)

- Phase 2 changes the action meaning, so old checkpoints will not transfer; start
  fresh after it.
- Phase 2 also changes human play: drag length becomes shot power.
- More games use more RAM; watch free memory with 6 running.
- Both-sides data can make self-play collapse into one style; keep the frozen pool
  and chaser in the mix.
- Warm start can lock in the chaser's style; raising `log_std` after copying guards
  against it.
- Lowering `MaxTurnSeconds` or adding damping changes how the game plays for humans
  too.
- On this Windows machine, Godot must be launched detached (`Win32_Process Create`),
  never as a child of the shell tool, or the shell hangs.


---

## 7. Phase 9-16: final state (addendum)

Sections 1-6 were written against the original Phase 0-8 plan. The plan was later
renumbered and extended to Phase 16; `docs/optimization_plan.md` is the source of
truth for the phases. This addendum records what the later phases added to the
code so the spec matches the shipped trainer. Where a "today" value in sections
1-6 differs from what shipped, the value here wins.

### 7.1 Phase-3 defaults as shipped

The entropy and KL defaults ended up different from the first draft in section
4.3. The shipped `training_loop.py` defaults are `--ent-coef 0.0` (the CleanRL
continuous default; exploration comes from `log_std`, not an entropy bonus),
`--target-kl 0.02`, `--minibatch-size 256`, `--n-passes 10`, `--max-grad-norm
0.5`. The PPO clip is now the `--clip-ratio` flag (default 0.2; self-play phases
used 0.1-0.15), not a hard-coded 0.2.

### 7.2 Phase 9: speed

Time scale raised (48 is the standard AI_VS_AI value) and the physics-steps cap
lifted so one Godot core is used fully; physics is unchanged because ticks scale.
No new learning flags.

### 7.3 Phases 10-11: live Elo, best gate, LayerNorm, critic

- **Live Elo (`live_elo.py`).** Scores the newest save WHILE training runs, every
  `--live-elo-freq` epochs, on ONE spare Godot at `--live-elo-shm`, against the
  fixed anchors chaser (1000) and defender (950), on the same anchored scale
  `league_eval` uses. Off when `--live-elo-shm` is unset. `--live-elo-games`,
  `--live-elo-sigma`, `--live-elo-step-timeout` and `--live-elo-check-budget` tune
  it; the short step timeout and the overall check budget were the Phase 12 review
  deadlock fix (a wedged spare game can no longer idle the run).
- **best.pt gate (`Best_tracker`).** A save becomes best when its live Elo clears
  the current best by MORE than `BEST_MARGIN_ELO` (25, the noise guard). The
  25-Elo margin is authoritative; the head-to-head vs the current best is folded
  in ONLY as a veto on a clear one-on-one loss (`loses_head_to_head`, score <
  0.45). Requiring the challenger to also WIN a noisy 20-game h2h (> 0.55) is what
  froze `best.pt` at epoch 0 for all 200 epochs of long run 2, so that second gate
  was removed. A drop of > `ROLLBACK_ELO_DROP` (100) Elo for `ROLLBACK_CHECKS` (2)
  checks rolls back to `best.pt`.
- **`--best-path` / `--checkpoint-dir`.** `best.pt` and the periodic
  `ppo_epoch_*.pt` saves are written under the run's OWN directory (default:
  `--checkpoint-dir` = the folder of `--best-path`), so two runs never overwrite
  each other's epoch saves. Each phase run lives in its own `p<N>_run/` folder.
- **Critic and self-play steps.** `--value-norm` normalizes value targets with a
  running mean/std saved in the checkpoint; `--n-passes-v` runs more critic passes
  than policy passes; `--clip-ratio` and `--n-minibatches` tune the update;
  `--kl-warm-start` / `--kl-coef` / `--kl-fade-fraction` add a mean-only KL loss
  toward a frozen warm start, faded over the first half.
- **LayerNorm (`--use-layernorm`, `--weight-decay`).** Adds LayerNorm to the
  hidden layers (Linear -> LayerNorm -> activation), which adds a parametrised
  layer and so changes the save format; a run needs a warm start rebuilt with it
  on. `PolicyOpponent.load` and `SelfPlayPool.load_state` auto-detect LayerNorm
  from the `pi_net.1.weight` key and rebuild the net to match, so a LayerNorm
  snapshot loads anywhere without a flag.

### 7.4 Phase 12: data and rewards (kept, default off)

Four flags, all defaulting OFF because on the 40-game T5 they did not beat the
Phase 11 recipe (the repeated-move exploit also scored worse), so a plain launch
reproduces Phase 11:
- `--input-norm`: normalize observations with a running mean/std inside the net,
  frozen at eval, saved in the checkpoint;
- `--random-start-prob`: fraction of resets that start from a random legal layout
  instead of kickoff (Godot places, Python decides which resets via byte 91; a
  random layout never puts the ball in a goal or a player in a wall);
- `--reward-scale`: scale rewards by the running std of the discounted return;
- `--potential-shaping`: exact potential-based shaping `gamma*phi(s') - phi(s)`
  with `phi = 0` at a goal and an out-of-bounds skip.

### 7.5 Phases 13 and 15: skipped on purpose

- **Phase 13 (bins)** was skipped: aim was already fine (action sigma ~0.19), so a
  discretized action head was not needed.
- **Phase 15 (exploiter / mirror loss / shared net)** was skipped: after long run
  2 the AI is no longer easy to beat, so the extra machinery was not justified.

### 7.6 Phase 14: long run 2, and the champion

A 200-epoch / ~1.2M-turn run from `best_p11_ln.pt` with the Phase 11 recipe. Its
epoch-100 save is the shipped champion `checkpoints/best_final.pt`: ~1156 Elo on
the 40-game T5, beating the Phase 11 best 27-13 head-to-head. `evaluate.py`
defaults to it (`default_model()` prefers `checkpoints/best_final.pt`, else the
newest `ppo_epoch_*.pt`).

### 7.7 File-map additions

| File | Phase(s) | Change |
|---|---|---|
| `bobble_model/live_elo.py` | 10-11 | new: live Elo + best.pt gate + rollback |
| `bobble_model/neural_network.py` | 11-12 | LayerNorm, value-norm, input-norm |
| `bobble_model/training_loop.py` | 9-12 | live-Elo wiring, run-dir saves, the Phase 11/12 flags |
| `bobble_model/evaluate.py` | 16 | `default_model()` -> `best_final.pt` |
| `scripts/archive/` | 16 | one-off per-phase scripts moved here for provenance |
| `README.md` | 16 | new: repo overview and final CLI |
