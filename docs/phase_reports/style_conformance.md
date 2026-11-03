# Style conformance: match Deacon's main-branch code style

Branch `ai-vs-ai-optimization` (worked at HEAD `fa971dc`). Goal: bring the code
style and comments on this branch into line with how Deacon writes code on
`main`, touching **only** files his feature work changed vs `main`
(`git diff main --stat`). Style and comments only -- no behaviour changes.

Decision on tone (from the user): keep his casual, plain-spoken register and
comment habits (explain the *why*, same casing and density), but **profanity-free
and in Canadian English**. Replace only the swear words; do not formalize the
rest. For C# indentation / braces where he is inconsistent, follow the style
already used in the file being edited -- no tab/space churn, no whole-file
reformat.

## (a) The style guide, from his own `main` commits

Source: commits by `Deacon Sham` on `main` (`git shortlog -sn main`), reading his
hand-written Python (`bobble_model/neural_network.py`,
`bobble_model/training_loop.py`, `bobble_model/shared_memory_link.py`,
`bobble_model/bobble_fuck_env.py`) and C# (`bobble_fuck/Scripts/GameManager.cs`,
`SharedMemory.cs`, `Ball.cs`, `Player.cs`). Agent commits on the feature branch
are **not** his style and were not used as the reference.

### Python

| Rule | What he does | `main` example |
|---|---|---|
| Indent | 4 spaces | all files |
| Strings | **single quotes**, incl. f-strings: `print(f'Using device: {device}')` | `training_loop.py` |
| Type hints | **none** -- no param or return annotations | `def update(ac, buf, pi_optimizer, ...)` `training_loop.py` |
| Docstrings | triple-quote block opening with `Docstring for <name>:` then a blank line and prose | every function in `neural_network.py` |
| Class names | `Capitalized_snake_case` (not PascalCase) | `Actor_critic_neural_network`, `PPO_buffer`, `Shared_memory_link` |
| Functions / vars | `snake_case` | `discount_cumsum`, `path_start_idx` |
| Constants | `UPPER_SNAKE`, module scope | `OBS_OFFSET`, `TOTAL_SIZE` in `shared_memory_link.py` |
| Comments | `#` inline, lower-case, explain the *why* not the *what*, moderate density | `# Normalize advantages: (A - mean(A)) / std(A)  # This helps with training stability` |
| Logging | `print(f'...')`, single quotes | `training_loop.py` |
| Errors | plain `raise TimeoutError(f"...")` / `assert` with a short message | `shared_memory_link.py` |
| Register | casual, first-person asides allowed (`# for mac users`, "thanks shawn"), never profane | `training_loop.py` |

### C#

| Rule | What he does | `main` example |
|---|---|---|
| Indent | 4 spaces in most files; **tabs in `Ball.cs`** -- inconsistent, so match the file | `GameManager.cs` (spaces) vs `Ball.cs` (tabs) |
| Braces | mixed: Allman for the class, K&R `{` on the same line for methods/blocks | `GameManager.cs`, `Player.cs` |
| Fields | `[Export]` attributes, `camelCase` private fields, `PascalCase` public | `GameManager.cs` |
| Comments | `//` lower-case, casual, explain the *why* | `// Mouse position on xy plane` |
| Logging | `GD.Print($"...")` | `SharedMemory.cs` |
| Register | on `main` it is often profane (`SCOREEEE...`, `*farts*`, `fucking`, name `BobbleFuckState`). Per the decision this register is kept **but profanity-free**. | -- |

## (b) Conformance audit and the files I will change

Audit finding, before any edit:

- **Profanity is already gone** from the current branch source. A word-boundary
  scan of every in-scope `.py` / `.cs` found no real swear words in names or
  comments: the branch already renamed `bobble_fuck/` -> `bobble_league/` and
  `BobbleFuckState` -> a clean name, and the old `SCOREEEE...` method is now
  `GameManager.Score(...)`. So "replace the swear words" is effectively done at
  the name level; nothing in scope reintroduces it.
- **Most branch code is already in his style**: `Capitalized_snake` classes
  (`Running_mean_std`, `Elo_table`, `Bobble_league_env`), the `Docstring for X:`
  opener, `UPPER_SNAKE` constants, no type hints, why-comments. The C# is clean,
  lower-case, casual, profanity-free.

The real, remaining drift is a handful of newer utility files that use
**double-quoted strings** and **short docstrings without his `Docstring for X:`
opener**, against his 100%-single-quote, opener-docstring `main` style. Those are
the only changes, and they are string/comment/docstring text only -- no code
logic, no identifiers, no public CLI flags.

Per-file plan (Python, `bobble_model/`):

- `power_probe.py` -- double quotes -> single quotes; add `Docstring for` opener
  to `player0_pos`, `shoot_at_power`; keep the casual lower-case wording.
- `evaluate.py` -- double quotes -> single quotes in code strings; align the
  function docstrings to his opener where one exists. Leave the module docstring
  usage block as-is (it is the `--help` text; wording unchanged).
- `launch_games.py`, `league_eval.py`, `live_elo.py`, `opponents.py`,
  `warm_start.py`, `bobble_league_env.py`, `training_loop.py`, `elo.py`,
  `neural_network.py`, `shared_memory_link.py` -- single-pass: any remaining
  double-quoted **code** string literals -> single quotes (his convention); add
  the `Docstring for X:` opener to any function/class docstring that is missing
  it; fix Canadian spelling in comments (`-ize` is fine; `color` -> `colour`,
  `behavior` -> `behaviour` in prose). No logic, no string *values* that cross
  the wire or appear in checkpoints/CLI, changed.
- `tests/` (`tests/mock/*.py`, `tests/unit/*.py`, `conftest.py`, `mock_godot.py`)
  -- same quote/docstring/spelling conformance, comments and test-local strings
  only. Assertion messages and any protocol/CLI literal left byte-identical.

Per-file plan (C#, `bobble_league/Scripts/`): `Ball.cs`, `DataManager.cs`,
`GameManager.cs`, `Player.cs`, `PlayerSpawn.cs`, `SharedMemory.cs`, `TeamGoal.cs`
-- already clean and in his register; only Canadian-spelling fixes in comments if
any, and no indentation/brace churn (match each file). Expect near-zero changes.

Commits, one area per commit, **by path only**:
1. `bobble_model` core (`neural_network`, `training_loop`, `shared_memory_link`)
2. `bobble_model` env + rewards (`bobble_league_env`, `elo`)
3. `bobble_model` eval/league (`evaluate`, `league_eval`, `live_elo`,
   `launch_games`, `opponents`, `power_probe`, `warm_start`)
4. C# scripts (`bobble_league/Scripts/*`)
5. tests (`tests/**`)

A commit is skipped if its pass produced no change.

## (c) What I will NOT touch

- **Behaviour**: no logic, control-flow, numeric, or algorithm changes. Verified
  style-only per area with `python -m compileall` + an AST comment/format compare.
- **Public contracts**: no renames of CLI flags, checkpoint keys, the 124-byte
  shared-memory protocol field names/offsets, or any string that crosses the wire
  or that an old checkpoint load depends on. `best_final.pt` must still load.
- **Identifiers** are renamed only when fully internal and every reference (code,
  tests, docs) is updated; the audit found none that need it, so none are planned.
- **Why-comments** that explain a non-obvious fix or decision are kept and only
  rewritten into his voice, never deleted.
- **Out-of-scope files** (left exactly as they are): the uncommitted
  `bobble_league/BobbleLeague.csproj` and `docs/optimization_plan.md`; untracked
  `bobble_model/p7_scratch/**` and `bobble_model/p7_eval.py`; every run folder;
  `.ps1` scripts (`scripts/**`, `bobble_model/p14_run/*.ps1`) -- run scaffolding,
  not his hand-written Python/C#; and all docs except this report.

## (d) Results

_(appended after Steps 3-4)_
