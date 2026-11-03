# Bobble League RL agent

An end-to-end reinforcement-learning setup for **Bobble League**, a 3-v-3,
physics-based, turn-based soccer game. A Godot 4 (C#/.NET) game plays the turns;
a Python (PyTorch) PPO trainer learns from them. The two talk over a small
file-backed shared-memory buffer, so one Python network call can drive many
headless Godot games at once.

Spelling across this repo is Canadian English (colour, centre, defence,
behaviour); `-ize` endings are kept. Engine and library API names are left as
they are.

All optimization work lives on the `ai-vs-ai-optimization` branch. The plan it
follows is `docs/optimization_plan.md`, phase by phase, with one report per phase
under `docs/phase_reports/`.

## The champion

The shipped model is `bobble_model/checkpoints/best_final.pt` — the Phase 14
"long run 2" best (epoch 100). On a 40-game-per-pairing T5 it scored ~1156 Elo,
beating the Phase 11 best (~1104) head-to-head 27-13, with win rates of ~56 % vs
the hard chaser and ~84 % vs the defender. `evaluate.py` loads it by default (AI
play and human play both), so you do not have to name a checkpoint.

## Layout

```
bobble_league/            Godot 4 game (C#/.NET)
  Scripts/                GameManager, DataManager, SharedMemory, Player, Ball, TeamGoal
  Scenes/                 scene.tscn (play), arena, player
bobble_model/             Python trainer and tools
  training_loop.py        PPO trainer + training CLI
  bobble_league_env.py    single + vector env, frames, reward shaping
  neural_network.py       Actor_critic_neural_network (MLP, optional LayerNorm/input-norm)
  opponents.py            still/random/chaser/defender/replay bots + self-play pool
  live_elo.py             live Elo + best.pt gate during training
  league_eval.py          offline Elo league (T5)
  launch_games.py         launch N detached headless Godots
  warm_start.py           behaviour-clone a warm start from the chaser
  evaluate.py             play / score a checkpoint; human play
  elo.py                  pure Elo maths
  power_probe.py          live check: power 0.2 moves less than 1.0 (Phase 2)
  checkpoints/            shipped models (best_final.pt, best_p11_ln.pt, ...)
docs/
  optimization_plan.md    the plan (source of truth for the phases)
  phase_reports/          one report per phase (phase0.md ... phase16.md)
  spec/                   requirements.md, design.md, tasks.md
scripts/
  smoke_test.ps1          T4 live smoke (build, import, short train, eval)
  proof_live_elo.ps1      a real live-Elo training burst
  archive/                one-off per-phase scripts, kept only for provenance
tests/
  unit/                   T1 unit tests (no Godot)
  mock/                   T2 tests against a mock Godot
  mock_godot.py           fake Godot speaking the shared-memory protocol
```

## Requirements

Python deps are in `bobble_model/requirements.txt` (PyTorch, NumPy, TensorBoard,
pytest). On the dev box PyTorch is the CUDA 12.6 (`cu126`) build so the GTX 1080
Ti works; CUDA barely helps here because Godot playing each turn is the
bottleneck, not the small network. The game needs Godot 4.6.1 .NET and the .NET
SDK.

Point the trainer at your Godot console binary with `--godot` or the
`BOBBLE_GODOT_EXE` environment variable (default is the dev box's Desktop build).

## Running

Run Python commands from `bobble_model/`.

### Train

```
# 6 games, launched here, from the shipped Phase 11 best, the Phase 11 recipe
python training_loop.py --games 6 --launch \
    --epochs 200 --steps-per-epoch 6000 \
    --init-checkpoint checkpoints/best_p11_ln.pt \
    --use-layernorm --value-norm --n-passes-v 2 --clip-ratio 0.15 \
    --best-path p_run/checkpoints/best.pt --checkpoint-dir p_run/checkpoints
```

Key flags (full list: `python training_loop.py -h`):

| Flag | What |
|---|---|
| `--games N` / `--launch` | run N parallel games; launch them here |
| `--epochs` / `--steps-per-epoch` | length; steps are the TOTAL across games |
| `--init-checkpoint` / `--resume` | start from a save / resume a full run |
| `--opponent` | `self` (pool), `chaser`, `random`, `still`, `none`, or a `.pt` |
| `--best-path` / `--checkpoint-dir` | where `best.pt` and the `ppo_epoch_*.pt` saves go (keep them in the run's own dir) |
| `--live-elo-shm` | shm path of one spare game to score live Elo on (off when unset) |
| `--use-layernorm` / `--value-norm` | LayerNorm hidden layers / running value-target norm (Phase 11) |
| `--pfsp` / `--shaping-fade` | prioritized self-play / fade shaping to 0 |

The Phase 12 data/reward flags (`--input-norm`, `--random-start-prob`,
`--reward-scale`, `--potential-shaping`) are **kept in the code but default off**:
on the 40-game T5 they did not beat the Phase 11 recipe, so a plain launch
reproduces Phase 11. See `docs/phase_reports/phase12.md`.

### Evaluate / score

```
python evaluate.py --opponent chaser --episodes 20        # best_final.pt vs the chaser
python evaluate.py --opponent checkpoints/best_p11_ln.pt  # vs another checkpoint
python league_eval.py --checkpoints checkpoints/best_final.pt checkpoints/best_p11_ln.pt \
    --games 40 --shm <spare-game-shm>                     # offline Elo league (T5)
```

With no `--checkpoint`, `evaluate.py` uses `checkpoints/best_final.pt`, falling
back to the newest `ppo_epoch_*.pt` if there is no final model yet.

### Play against the AI (human vs AI)

1. Open `bobble_league` in the Godot editor, open `Scenes/scene.tscn`, and set
   the `GameManager`'s **CurrentPlayMode** to `HUMAN_VS_AI`. Run the scene (you
   are Team A: left-drag to aim, drag length sets power, right-click ends your
   turn).
2. In another terminal, from `bobble_model/`:

   ```
   python evaluate.py --human
   ```

   This waits for you with no turn timeout and plays `best_final.pt` with a
   shrunk sampling spread (std × 0.3) so one fixed counter does not always work.
   Human turns settle at the slower `HumanStopSpeed` (0.3), and a turn that never
   settles ends after `MaxTurnSeconds` (10 s).

#### The AI's Elo

Both sides show the model's measured Elo so you know how strong the opponent is.
`evaluate.py` prints it at start (`Elo: 1104`), and in `HUMAN_VS_AI` the Godot
window shows **AI Elo 1104** in the top-right corner.

The number comes from a small sidecar next to the checkpoint,
`checkpoints/best_final.elo.json` (`{"elo": 1104, ...}`), so re-scoring a model
only rewrites that one-line file, not the 7 MB save, and Godot reads the same
file without touching the shared-memory protocol. `best_final.pt`'s 1104 is its
Phase 16 T5 score (40 games a pairing, `bobble_model/p16_t5/logs/elo.csv`). If
the sidecar is missing an `elo` entry can instead live inside the checkpoint
dict; with neither, both sides show **unrated**.

## Testing

| Level | Command | Godot? |
|---|---|---|
| T1 unit | `pytest -m "not live" tests/unit` | no |
| T2 mock | `pytest -m "not live" tests/mock` | no |
| all fast | `python -m pytest -m "not live"` | no |
| T4 smoke | `scripts/smoke_test.ps1` | yes |
| T5 strength | `league_eval.py` on real games | yes |

Python, NumPy and PyTorch are seeded in tests so T1/T2 repeat exactly. Godot
physics is not fully repeatable, so the live levels pass on ranges, not exact
numbers. A bug found by hand gets a test before it is fixed.

## Shared-memory protocol

The Python and Godot sides share a fixed 120-byte mmap, versioned by a byte so a
mismatched build is refused. The protocol is at **version 5** (angle+power actions
since v3, turn-length telemetry added in v4, random-start / out-of-bounds flags in
v5). A checkpoint saved at action format ≥ 3 still loads; a pre-angle+power save
is refused. Layout and handshake are in `docs/spec/design.md`.
