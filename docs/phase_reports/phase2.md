# Phase 2 report: angle + power action (protocol v3)

Date: 2026-10-31. Branch: `ai-vs-ai-optimization`. Builds on Phase 1.

## Commits

Split into two commits on `ai-vs-ai-optimization`, so the Phase 2 commit shows
only the angle+power work:

1. **"carry over game manager fixes"** — `82ca3c2`. Only `GameManager.cs`, carrying
   the prior-session fixes that were never committed (the Phase 1 report left them
   out): `ReadCommandLine`, `MaxTurnSeconds` + the 20 s turn timeout, the
   out-of-bounds-to-kickoff reset, `SetTimeScale`, and the field-geometry exports.
   The live smoke depends on these, and this commit builds on its own
   (`dotnet build` 0 errors).
2. **"phase 2: angle and power action"** — the newest commit on the branch. Only
   the angle+power work on top of commit 1 (`GameManager.cs` now just the
   `ApplyTeamAction` decode + the human-drag power line). This report is folded in
   via `git commit --amend`, so its own final hash cannot be printed inside it;
   `git log -1` shows it. The pre-amend hash was `5fd5180`.

The final tree is byte-identical to the earlier single commit `1352c28`
(`git diff 1352c28 HEAD` shows only this report), so the live smoke below was not
re-run after the split.

Staged by explicit path only. Left out on purpose, as Phase 0/1 did: the other
prior-session working-tree edits (`TeamGoal.cs`, `project.godot`,
`optimization_plan.md`, `export_presets.cfg`, the deleted `runs/ppo_experiment`
tfevents), all generated artifacts (`checkpoints/`, `runs/`, `old_runs/`, `logs/`,
the junk `BobbleLeague.csproj.old*`), and the untracked spec files
(`docs/spec/requirements.md`, `design.md`, `docs/phase_reports/phase0.md`). Only
`docs/spec/tasks.md` was committed, to tick the Phase 2 boxes. Nothing pushed.

## The invariant that makes it correct

For both teams, an action decoded by Python's mirror and by Godot's decode
launches the player in the **same world direction**, and power 0.2 moves a player
clearly less than power 1.0.

- Angle `a` in [-1, 1] -> `theta = a * pi` measured from -z (the enemy goal), so
  `a = 0` aims at the enemy goal and the `a = +-1` **seam** points straight back
  at own goal (+z) -- the move a player should need least. Direction is
  `(sin theta, -cos theta)` in the team's mirrored frame, matching Godot's
  `Atan2(x, z)`.
- Power `p` in [-1, 1] -> `0.2 + 0.8 * (p + 1) / 2`, so shots run 20% to 100%.
- Godot's `ApplyTeamAction` applies `FrameZ` (team 2 -> z flipped) to the decoded
  z, so team 1 and team 2 sending the same mirrored-frame action launch mirror
  world directions -- the same relationship the obs frames are built on.

## What changed

**Python (`bobble_model/`).**
- `bobble_league_env.py`: new `decode_angle`, `decode_power`, `decode_action`,
  `encode_action`, `vector_to_action` share one definition for the env, the
  chaser, the probe and the tests. The env passes the 6 wire numbers through;
  Godot does the decode.
- `opponents.py`: `ChaserOpponent` builds its aim vectors as before, then packs
  them with `vector_to_action`. Kickoff non-movers now aim back at own goal at the
  0.2 power floor (v3 has no zero-power "don't move"), a harmless nudge away from
  the ball instead of a weak shot at the enemy goal. `load_state_dict` refuses a
  pre-Phase-2 checkpoint with a clear message and reads full saves with
  `weights_only=False`.
- `training_loop.py`: checkpoints carry `action_format` (= the protocol version);
  `load_checkpoint` refuses an untagged/old save rather than loading garbage,
  since the tensor shapes did not change.
- `shared_memory_link.py`: `PROTOCOL_VERSION` 2 -> 3; layout comment updated to the
  angle+power action.
- `power_probe.py` (new): live check of the power 0.2 vs 1.0 move distance.

**Godot (`bobble_league/Scripts/`).**
- `Player.cs`: `Launch(powerScale = -1)` scales launch force by the power instead
  of always normalizing; a `launchPower` field carries the per-player power so
  every `Launch()` call site uses it, reset to 1 after firing.
- `GameManager.cs`: `ApplyTeamAction` decodes angle+power into the aim `input` and
  `launchPower`, `FrameZ` on z; human drag length now sets `launchPower` (clamped
  to [0.2, 1]).
- `SharedMemory.cs`: `PROTOCOL_VERSION` 2 -> 3; layout comment updated.

**Mock + tests.** `tests/mock_godot.py` speaks v3 and decodes angle+power so fake
physics still advances. New `tests/unit/test_actions.py`; mirror round-trips added
to `test_frames.py`; v2+v3 version checks in `test_protocol.py`; old-checkpoint
refusal tests in `test_checkpoint.py`; `test_evaluate.py` now saves a tagged
checkpoint.

## Nearby cases tested, and where

| Case | Covered by |
|---|---|
| angle decode at -1, 0, +1 | `test_actions::test_angle_at_edges_points_where_expected` |
| angle at the **seam** ±1 is the same direction (own goal) | `test_actions::test_seam_is_at_own_goal` |
| angle at 0.5 / -0.5 (straight across +x / -x) | `test_actions::test_angle_at_edges_points_where_expected` |
| power decode at -1, 0, +1; stays in [0.2, 1] | `test_actions::test_power_at_edges`, `test_power_stays_in_range` |
| power 0.2 is clearly under 1.0 (unit) | `test_actions::test_power_0p2_is_clearly_less_than_1p0` |
| decoded aim is unit length for all angles | `test_actions::test_decoded_direction_is_unit_length` |
| full 6-number action shape + scaling | `test_actions::test_decode_action_shape_and_scaling` |
| chaser/human vector -> action -> decode round trip | `test_actions::test_vector_to_action_round_trips_through_decode`, `test_frames::test_action_vector_round_trip` |
| **both teams** launch the same world direction (z flip) | `test_actions::test_both_teams_same_world_direction`, `test_mirrored_board_same_world_shot` |
| action mirror flips z like the board | `test_frames::test_action_mirror_flips_z_like_the_board` |
| protocol version is 3 | `test_protocol::test_version_is_v3` |
| old Godot refused: v1 and **v2** buffers | `test_protocol::test_old_version_is_refused`, `test_v2_buffer_is_refused` |
| old checkpoint refused (bare + untagged), both loader paths | `test_checkpoint::test_old_checkpoint_is_refused_*` |
| new tagged checkpoint loads in both paths | `test_checkpoint::test_new_checkpoint_loads_through_both_paths` |
| **power -1 and +1** move distance on real physics | live power probe (below) |
| **human drag short vs long** -> power | Godot `_Process` sets `launchPower` from drag length; not unit-tested (needs mouse input + real physics), the AI power path proves the same `launchPower` -> `Launch` scaling live |

The human drag -> power path has no unit test because it needs real mouse input
and Jolt physics; it uses the exact same `launchPower` -> `Player.Launch` scaling
the live power probe exercises through the AI action, so the scaling itself is
covered live.

## Results

- `pytest -m "not live"`: **58 passed** (42 from Phase 1 + 16 new/changed for
  Phase 2).
- `dotnet build`: **Build succeeded, 0 Error(s)** (one pre-existing
  `AnimatedTexture` deprecation warning in `GameManager.cs`, not from this phase).
- Godot `--import`: succeeded; the csproj SDK bump the import makes is reverted by
  the smoke script (verified: `BobbleLeague.csproj` unchanged in git).
- **T4 live smoke** (`scripts/smoke_test_phase2.ps1 -Games 6`, 2 epochs x 2400
  total turns, timescale 32, 10-game eval of the fresh checkpoint):

  | metric | value |
  |---|---|
  | games | 6 |
  | total turns | 2400 in 1.45 min |
  | turns/min | 1656 |
  | goals (W+L) | 217 |
  | stuck turns | 0 |
  | restarts | 0 |
  | out of bounds | 13 (6% of games) |
  | python tracebacks | 0 |

  **SMOKE PASS.**

- **Power 0.2 vs 1.0** (live, `power_probe.py --turns 6`, real engine):

  | power | mean move | per-shot |
  |---|---|---|
  | 0.2 | **0.375** | [2.237, 0, 0, 0, 0.001, 0.01] |
  | 1.0 | **7.185** | [7.065, 7.065, 7.027, 7.039, 6.572, 8.344] |

  Full power moves **~19x** as far as min power -- power 0.2 clearly moves a player
  less than 1.0. (The one non-zero 0.2 reading, 2.237, is the very first shot
  before the player had settled; the settled 0.2 shots move ~0.)

## Free RAM check

Before the 6-game smoke: **4920 MB free of 16334 MB** (well above the ~1.5 GB
floor), so the smoke ran with 6 games, not the 4-game fallback.

## Notes for later phases

- Old checkpoints no longer apply (the action meaning changed). The old
  `checkpoints/ppo_epoch_199.pt` and friends are now refused with a clear message;
  the fresh-start risk the plan called out is enforced, not just documented.
  Training the next run (Phase 3) starts from scratch.
- `evaluate.py` with no `--checkpoint` still globs the newest `ppo_epoch_*.pt`,
  which may be a stale pre-Phase-2 file; it will refuse it with the clear message.
  Pass `--checkpoint` to a fresh file, or archive the old ones, when evaluating.
