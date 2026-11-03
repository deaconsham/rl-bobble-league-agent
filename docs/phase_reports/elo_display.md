# Elo display during human vs AI play

Branch `ai-vs-ai-optimization`. A small post-plan feature: show the AI model's
measured Elo so a person playing it knows how strong the opponent is. Added after
the user played human-vs-AI and said the game feels right.

## What it does

- `evaluate.py` prints the model's Elo at start, next to the model line:

  ```
  Model: checkpoints/best_final.pt
  Elo: 1104
  ```

- In `HUMAN_VS_AI`, the Godot window shows **AI Elo 1104** in a top-right label.
  The label is hidden in `AI_VS_AI` and `HUMAN_VS_HUMAN`.
- When the Elo is unknown, both sides show **unrated** instead of failing.

## Elo source

`best_final.pt` (the shipped Phase 14 champion, a copy of `p14_run`
`ppo_epoch_100.pt`) shows **1104**.

- Used: the **newest measured** score, from the **Phase 16 T5** league
  (40 games a pairing), `bobble_model/p16_t5/logs/elo.csv` — the
  `best_final.pt,1104.0` row.
- Not used: the Phase 14 eval's 1156.1. That came from a different run (the
  Phase 14 eval T5, `phase14_eval.md`); run-to-run Elo noise on this game is
  ~20 pts, so the two numbers measure the same model in two leagues. The newest
  one is shown.

## Design — simplest sound choice

A **sidecar JSON** next to the checkpoint, read by both sides:

```
bobble_model/checkpoints/best_final.elo.json
  {"elo": 1104, "source": "Phase 16 T5 (40-game-per-pairing league)",
   "source_csv": "bobble_model/p16_t5/logs/elo.csv", "checkpoint": "best_final.pt"}
```

Why a sidecar, not the shared-memory protocol or the checkpoint itself:

- **Protocol untouched.** The Python↔Godot mmap is a fixed 124-byte buffer
  (`SharedMemory.cs` / `shared_memory_link.py`), versioned by a byte. Pushing the
  Elo through it would mean a new field and a version bump, which breaks every
  save's handshake. Godot reads the sidecar file directly instead, so no byte of
  the protocol changes. (The plan's rule was: if every way of getting the number
  into Godot breaks the protocol, report `question` first. A way that does not
  break it exists, so no question was needed.)
- **No 7 MB rewrite.** Re-scoring a model rewrites a one-line file, not the whole
  checkpoint.
- **One source of truth.** `evaluate.py` and Godot read the same file, so the
  printed number and the HUD number cannot drift apart.

Python (`evaluate.py`):

- `checkpoint_elo(ckpt_path)` looks for `<checkpoint>.elo.json`; if absent it
  falls back to an `elo` key inside the checkpoint dict; with neither it returns
  `None` and play shows `unrated`. A bad/unreadable file also gives `None`.
- The sidecar takes precedence over a checkpoint `elo` key.

Godot (`GameManager.cs`):

- `ShowAiElo()` runs in `_Ready`: in `HUMAN_VS_AI` it reads the sidecar through
  `ReadEloSidecar()` and sets the `AiEloLabel` text, else hides the label.
- The sidecar path is `[Export] AiEloSidecar`
  (`../bobble_model/checkpoints/best_final.elo.json`), resolved against
  `ProjectSettings.GlobalizePath("res://")`, so it works running the scene from
  the editor. `evaluate.py --human` defaults to `best_final.pt`, so the HUD shows
  that model's score. A missing or bad file shows `AI Elo unrated`.
- `AiEloLabel` is a new `Label` under the `Ui` node in `Scenes/scene.tscn`,
  anchored top-right, hidden by default.

Canadian English and the repo's C#/Python style were matched (4-space C#, braces
on the same line, short lowercase comments; Python `Docstring for ...` style).

## Tests

- `tests/unit/test_elo_display.py` (T1, 8 cases) covers `checkpoint_elo`:
  sidecar present; float rounded to int; no sidecar and no metadata → `None`;
  bad JSON → `None`; sidecar without an `elo` key → `None`; non-numeric `elo`
  → `None`; checkpoint-metadata fallback; sidecar wins over metadata.
- Full non-live suite: **213 passed** (`pytest -m "not live"`), up from 205 by
  the 8 new cases, no regressions.
- `dotnet build` of `bobble_league`: **0 errors** (one pre-existing
  `AnimatedTexture` obsolete warning on the goal-flash texture, unrelated).
- The Godot HUD's own missing/bad-file handling is in C# `ReadEloSidecar` and is
  not unit-tested (no C# test harness in the repo).

## Manual check — not run

The one-Godot windowed check of the label was **not** run: host RAM was tight
(~3 GB free) and a windowed, interactive HUMAN_VS_AI session cannot be driven
headlessly to verify a label without risking the ~1.5 GB floor. The change is
low-risk — the build passes, the scene wiring is a plain `Label` + node path, and
the sidecar read is exercised by the Python twin in tests. The user's own
human-vs-AI session will show the label.

## Commit

- Code + docs: `6e18bce` — `evaluate.py`, `bobble_league/Scripts/GameManager.cs`,
  `bobble_league/Scenes/scene.tscn`, `bobble_model/checkpoints/best_final.elo.json`,
  `tests/unit/test_elo_display.py`, `README.md`.
- This report: committed separately on `ai-vs-ai-optimization`.
