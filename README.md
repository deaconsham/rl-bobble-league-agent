# Bobble League RL Agent

A reinforcement-learning setup for **Bobble League**, a 3-v-3, physics-based,
turn-based soccer game. A Godot 4 game plays the turns; a PyTorch PPO trainer
learns to play them well. The two sides talk over a small shared-memory buffer,
so one Python network call can drive many headless games at once.

The shipped model (`bobble_model/checkpoints/best_final.pt`) beats every test
bot: ~56 % vs the hard chaser, ~84 % vs the defender, and it wins the offline
Elo league against the previous best.

## How it works

**Self-play PPO.** One policy plays both teams. It learns by playing games,
keeping the moves that led to goals and dropping the ones that did not, over and
over, for millions of turns.

**Shaped rewards.** Goals are rare, so the agent also gets small hints each turn:
move the ball toward the enemy net, keep a player between the ball and your own
net. These are tuned in Python so the game does not need rebuilding to change
them.

**Opponent pool.** As training goes, snapshots of past selves are saved into a
pool. The agent practises against a mix of its old selves plus scripted bots
(chaser, defender, a move-repeater), so it does not overfit to one opponent.

**Live Elo gate.** Every few epochs the current policy plays scoring games to get
an Elo rating. The best-rated checkpoint so far is saved as `best.pt`. Elo picks
the champion; it is not the training reward (see "How the model works").

**Parallel games.** The trainer launches N headless Godot games at once and steps
them together through one shared-memory buffer, so the small network stays busy
and training is not stuck waiting on one game.

## Stack

- **Game:** Godot 4.6.1 (.NET / C#) — physics, turns, rendering
- **Trainer:** Python + PyTorch (PPO, GAE, self-play league)
- **Bridge:** a fixed 120-byte memory-mapped file (versioned protocol)
- **Logging:** TensorBoard
- **Tests:** pytest (unit + mock-Godot), PowerShell smoke scripts for live checks

## Setup

### 1. Game

Install Godot 4.6.1 .NET and the .NET SDK. Point the trainer at your Godot
console binary with `--godot` or the `BOBBLE_GODOT_EXE` environment variable.

### 2. Trainer

Install dependencies from `bobble_model/`:

```
pip install -r requirements.txt
```

On the dev box PyTorch is the CUDA 12.6 (`cu126`) build so the GTX 1080 Ti
works. CUDA barely helps here — Godot playing each turn is the bottleneck, not
the small network.

### 3. Run it

Run all Python commands from `bobble_model/`.

**Play against the AI (human vs AI).** In the Godot editor, open
`Scenes/scene.tscn`, click the `GameManager` node, set **CurrentPlayMode** to
`HUMAN_VS_AI`, and press Play. You are Team A: left-drag to aim, drag length sets
power, right-click ends your turn. Then in a terminal:

```
python evaluate.py --human
```

This plays `best_final.pt` with no turn clock for you.

**Watch the AI play a bot.**

```
python evaluate.py --opponent chaser --episodes 20
```

**Train from the shipped best, 6 games at once.**

```
python training_loop.py --games 6 --launch \
    --epochs 200 --steps-per-epoch 6000 \
    --init-checkpoint checkpoints/best_p11_ln.pt \
    --use-layernorm --value-norm --n-passes-v 2 --clip-ratio 0.15 \
    --best-path p_run/checkpoints/best.pt --checkpoint-dir p_run/checkpoints
```

Each game uses ~170 MB of RAM; drop to `--games 4` if free RAM is under 1.5 GB.
Full flag list: `python training_loop.py -h`.

**Score two checkpoints in an Elo league (T5).**

```
python league_eval.py \
    --checkpoints checkpoints/best_final.pt checkpoints/best_p11_ln.pt \
    --games 40 --shm <spare-game-shm>
```

## How the model works

The agent is a small actor-critic network trained with PPO (Proximal Policy
Optimization).

**What it sees and does.** Each turn the network reads a short observation vector
(ball and player positions, team, phase) and outputs an **angle** and a **power**
for the shot. The actor head picks the shot; the critic head guesses how good the
current board is, which is the yardstick PPO measures each move against.

**How it learns.** PPO plays a batch of turns, works out which moves did better
than the critic expected (the advantage, via GAE), then nudges the policy toward
the better moves — but only a little per step, with a clip that stops any one
update from swinging the policy too far. This "small, safe steps" rule is what
keeps PPO stable (Schulman et al. 2017).

**Why Elo is a gate, not the reward.** Elo is one number per *game*, measured
against other players. PPO needs a signal on every *move*, cheap to compute, that
depends only on what the agent did. Elo is none of those — too rare, too slow,
too dependent on the opponent. So the reward is goals plus the shaping hints,
and Elo is used only to *pick the champion* and *choose which past selves to
practise against*. Think of the reward as your homework marks (every question,
right away) and Elo as your class rank (end of term, depends on classmates): you
study from the marks, you check the rank.

**Extras that made it learn better:**

- **LayerNorm** on the hidden layers, which steadies learning when the opponent
  keeps changing (as it does in self-play).
- **Value / return normalization**, so the critic's targets stay on a sane scale
  (value normalization "often helps and never hurts" in the MAPPO study).
- **Orthogonal init** and a **tight starting aim spread**, both from the widely
  used PPO implementation-details guidance (Andrychowicz et al. 2020).

## Layout

```
bobble_league/        Godot 4 game (C#/.NET)
  Scripts/            GameManager, DataManager, SharedMemory, Player, Ball, TeamGoal
  Scenes/             scene.tscn (play), arena, player
bobble_model/         Python trainer and tools
  training_loop.py    PPO trainer + training CLI
  bobble_league_env.py  single + vector env, frames, reward shaping
  neural_network.py   actor-critic network (optional LayerNorm / input-norm)
  opponents.py        scripted bots + self-play pool
  live_elo.py         live Elo + best.pt gate during training
  league_eval.py      offline Elo league (T5)
  launch_games.py     launch N detached headless Godots
  warm_start.py       behaviour-clone a warm start from the chaser
  evaluate.py         play / score a checkpoint; human play
  elo.py              pure Elo maths
  checkpoints/        shipped models (best_final.pt, best_p11_ln.pt, ...)
docs/
  optimization_plan.md  the plan (source of truth for the phases)
  phase_reports/      one report per phase (phase0.md ... phase16.md)
  spec/               requirements.md, design.md, tasks.md
scripts/
  smoke_test.ps1      T4 live smoke (build, import, short train, eval)
  proof_live_elo.ps1  a real live-Elo training burst
  archive/            one-off per-phase scripts, kept only for provenance
tests/
  unit/               unit tests (no Godot)
  mock/               tests against a mock Godot
  mock_godot.py       fake Godot speaking the shared-memory protocol
```

## Development

The training loop lives in `bobble_model/training_loop.py`; the reward shaping
and environment are in `bobble_league_env.py`; the network is in
`neural_network.py`. The Phase 12 data/reward flags (`--input-norm`,
`--random-start-prob`, `--reward-scale`, `--potential-shaping`) are kept in the
code but default off — on the 40-game T5 they did not beat the Phase 11 recipe.

Spelling across the repo is Canadian English (colour, centre, defence); engine
and library API names are left as they are. A bug found by hand gets a test
before it is fixed.

### Testing

| Level | Command | Godot? |
|---|---|---|
| unit | `pytest -m "not live" tests/unit` | no |
| mock | `pytest -m "not live" tests/mock` | no |
| all fast | `python -m pytest -m "not live"` | no |
| T4 smoke | `scripts/smoke_test.ps1` | yes |
| T5 strength | `league_eval.py` on real games | yes |

Python, NumPy and PyTorch are seeded so the unit/mock tests repeat exactly.
Godot physics is not fully repeatable, so the live levels pass on ranges, not
exact numbers.

### Shared-memory protocol

The Python and Godot sides share a fixed 120-byte mmap, versioned by a byte so a
mismatched build is refused. The protocol is at **version 5** (angle + power
actions since v3, turn-length telemetry in v4, random-start / out-of-bounds flags
in v5). Layout and handshake are in `docs/spec/design.md`.

## References

The model follows standard PPO practice. The papers and write-ups it draws on:

- **PPO:** Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy
  Optimization Algorithms* (2017). [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- **GAE:** Schulman, Moritz, Levine, Jordan, Abbeel. *High-Dimensional Continuous
  Control Using Generalized Advantage Estimation* (2015). [arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)
- **PPO implementation details:** Andrychowicz et al. *What Matters in On-Policy
  Reinforcement Learning?* (2020). [arxiv.org/abs/2006.05990](https://arxiv.org/abs/2006.05990)
- **The 37 Implementation Details of PPO:** Huang et al., ICLR Blog (2022).
  [iclr-blog-track.github.io/2022/03/25/ppo-implementation-details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- **Value normalization (MAPPO):** Yu et al. *The Surprising Effectiveness of
  PPO in Cooperative Multi-Agent Games* (2021). [arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955)
- **Critic passes (PPG):** Cobbe et al. *Phasic Policy Gradient* (2020).
  [arxiv.org/abs/2009.04416](https://arxiv.org/abs/2009.04416)
- **Warm start (Kickstarting):** Schmitt et al. *Kickstarting Deep Reinforcement
  Learning* (2018). [arxiv.org/abs/1803.03835](https://arxiv.org/abs/1803.03835)
