# Phase 0 report: cleanup + test harness

Date: 2026-10-31. Branch: `ai-vs-ai-optimization`.

## Commit

`2842a8c` — "phase 0: cleanup and test harness"
(full: `2842a8c2b06b323a50eca0a4e1d535dbc885d838`)

Staged by explicit path. `docs/spec/`, `docs/optimization_plan.md` and
`export_presets.cfg` were deliberately left out of the commit.

## What changed

**Renames finished.** Every Python reference to the old names is gone:
`bobble_fuck_env` -> `bobble_league_env`, class `Bobble_fuck_env` ->
`Bobble_league_env`, helper `bobble_fuck_env_helper` -> `bobble_league_env_helper`,
and the `bobble_fuck/...` paths in docstrings now read `bobble_league/...`.
Imports fixed in `opponents.py`, `evaluate.py`, `training_loop.py`.

**Comments cleaned.** Removed the ASCII-art block and the `godgod...` line from
`DataManager.cs` (kept the real `SharedMemory memory;` field that lived inside
it, and dropped a stray UTF-8 BOM the edit introduced). The owner's casual
lowercase comment style is kept everywhere.

**Canadian English.** `defense` -> `defence` in `RewardConfig` and its two uses.
`-ize` kept. Engine API names (`albedo_color`, `blender/meshes/colors`, ...)
left untouched — they are not ours to rename.

**Test harness added.**
- `tests/mock_godot.py` — fake Godot speaking protocol v2 on a shm file; can be
  told to hang (`set_hang`) or die (`die`).
- `tests/conftest.py` — puts `bobble_model` on the path, seeds python/numpy/torch.
- `pytest.ini` — registers the `live` marker so `-m "not live"` runs T1+T2 only.
- T1 unit (`tests/unit/`): `test_protocol.py`, `test_frames.py`,
  `test_rewards.py`, `test_gae.py`.
- T2 mock (`tests/mock/`): `test_single_game.py`, `test_evaluate.py`.
- `scripts/smoke_test.ps1` — T4 live smoke (build, import + csproj-bump revert,
  detached Godot, short train, 10-game eval, number checks, stops only its PID).
- `pytest==9.2.0` added to `bobble_model/requirements.txt`.

## Nearby inputs tested

| Input / case | Covered by |
|---|---|
| Team A (learner) scores -> +1, episode ends | `test_single_game`, `test_evaluate`, live smoke |
| Team B (opponent) scores -> -1, episode ends | `test_single_game` |
| reset after a goal (kickoff, next episode) | `test_single_game`, `test_evaluate` |
| step hang / timeout -> TimeoutError | `test_single_game`; live smoke (0 stuck) |
| old-protocol buffer refused | `test_protocol`, `test_single_game` |
| reward shaping terms (ball->goal, players->ball, defence, time) | `test_rewards` |
| GAE terminal (0 bootstrap) vs truncated (critic bootstrap) | `test_gae` |
| frame mirror involution / kickoff symmetry | `test_frames` |
| out-of-bounds -> kickoff, no reward | live smoke only (needs real physics; 4 handled, 0 stuck) |

Out-of-bounds has no unit test because it only arises from real Jolt physics,
not the mock's simple drift. The live smoke exercises it instead.

## Results

- `pytest -m "not live"`: **29 passed in 7.44s**
- `dotnet build`: **Build succeeded, 0 Error(s)** (1 pre-existing `AnimatedTexture`
  deprecation warning in `GameManager.cs`, not introduced here)
- Godot `--import`: succeeded; csproj SDK stayed `Godot.NET.Sdk/4.6.0` (no bump to
  revert)
- T4 live smoke (`scripts/smoke_test.ps1`, 2 epochs x 300 steps, 10-game eval,
  timescale 16): **goals = 51 (W 19 / L 32), stuck turns = 0, out of bounds = 4
  (8% of 51 games), python tracebacks = 0 — SMOKE PASS**
