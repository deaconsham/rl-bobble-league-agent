# Elo display removed

Branch `ai-vs-ai-optimization`. The AI-Elo display added in commits `6e18bce`
(code) and `ef045b9` (the `elo_display.md` report) was removed again at the
user's request. The original report is kept as history with a "Removed" note at
its top; this file records what was taken out.

## What was removed

Done as a targeted reversal of `6e18bce` (not a blind `git revert`), so only the
Elo-display pieces were touched. The pre-feature file blobs were restored exactly.

- **`bobble_model/evaluate.py`** — dropped `import json`, the `checkpoint_elo()`
  helper, and the two `Elo: <n>` print lines in `main()`.
- **`bobble_league/Scripts/GameManager.cs`** — dropped the `System.IO` and
  `System.Text.Json` usings, the `[Export] Label aiEloLabel` and
  `[Export] string AiEloSidecar` fields, the `ShowAiElo()` call in `_Ready`, and
  the `ShowAiElo()` / `ReadEloSidecar()` methods.
- **`bobble_league/Scenes/scene.tscn`** — dropped the `AiEloLabel` Label node, its
  `aiEloLabel = NodePath(...)` assignment, and the `"aiEloLabel"` entry in the
  `GameManager` `node_paths` array.
- **`bobble_model/checkpoints/best_final.elo.json`** — deleted (the sidecar).
- **`tests/unit/test_elo_display.py`** — deleted (8 cases).
- **`README.md`** — removed the "The AI's Elo" human-play subsection.

## Not touched

The 124-byte shared-memory protocol, `elo.py`, `league_eval.py`, and every
checkpoint other than the deleted sidecar json. The pre-existing unrelated
working-tree changes (`BobbleLeague.csproj`, `docs/optimization_plan.md`, the
untracked `.pt`/run artifacts) were left out of this commit.

## Verification

- Non-live pytest: `python -m pytest -m "not live" -q` → **205 passed** (back to
  the pre-feature count; the Elo suite was the only test added by the feature).
- `dotnet build` (bobble_league): **0 errors**. The one `CS0618 AnimatedTexture`
  warning is pre-existing and unrelated.
- `grep` of the repo for `elo.json`, `checkpoint_elo`, `AI Elo`, `aiEloLabel`,
  `AiEloSidecar`, `ReadEloSidecar`, `ShowAiElo` → matches only inside the kept
  historical `elo_display.md`; none in code, scene, tests, or README.

## Commit

Removal commit hash: recorded in the worker report (committed by path on
`ai-vs-ai-optimization`, not pushed).
