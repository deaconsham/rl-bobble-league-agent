# Phase 14 fixes: best.pt gate that never promoted, and epoch saves to the shared dir

Branch `ai-vs-ai-optimization`. Fix commit **`eb82e5a`** (code + tests); this
report is a separate commit on top.

Two bugs seen in Phase 14 "long run 2" (log:
`bobble_model/p14_run/logs/train.log`, launch:
`docs/phase_reports/phase14_started.md`). Both are fixed, with no-Godot regression
tests for each plus a real-Godot smoke.

---

## Bug A — `best_p14.pt` never promoted past epoch 0

### Symptom

`best_p14.pt` stayed at the epoch-0 save (live Elo 961.5) for all 200 epochs,
even though several later checks beat it by far more than the 25-Elo margin:

```
Live Elo @ 0:   mean 961.5  -> first best at epoch 0, elo 961.5
Live Elo @ 100: mean 1203.1  best 961.5 -> epoch 100: elo 1203.1, best 961.5, down streak 0
Live Elo @ 130: mean 1102.2  best 961.5 -> epoch 130: elo 1102.2, best 961.5, down streak 0
Live Elo @ 199: mean 1025.5  best 961.5 -> epoch 199: elo 1025.5, best 961.5, down streak 0
```

Every post-epoch-0 line ends in the tracker's *fallback* reason
(`"epoch N: elo ..., best ..., down streak 0"`), never `"new best ..."` — so the
promotion branch in `Best_tracker.consider` was never taken.

### Root cause

The best-gate in `bobble_model/live_elo.py` required a challenger to pass **two**
gates to replace `best.pt`:

1. clear the current best by more than `BEST_MARGIN_ELO` (25) on the anchor scale
   (the Phase 11 noise guard, commit `66f1e0e`), **and**
2. **win** the head-to-head vs the current best with `beats_best(...) > 0.55`.

Gate 1 passed at epochs 100/130/199. Gate 2 is the problem: it is a *noisy
20-game* head-to-head, and requiring the challenger to win it >55% vetoes a save
that is far ahead on the anchor scale but only level one-on-one in a small sample.
Replaying the exact logged mean-Elo sequence through `Best_tracker` with a level
h2h (`10-10-0`, score 0.50) reproduces the frozen-best behaviour exactly:
`beats_best(.., 0.50) == False`, so no promotion — for all 200 epochs.

The 25-Elo margin was *added* in `66f1e0e` specifically as the noise guard ("above
the noise band"), which makes the >55% h2h win a redundant **second** noise gate
— and the one that never opened.

### Fix

`bobble_model/live_elo.py`: the margin is now the authoritative gate. The h2h is
folded in only as a veto against a *false positive* — a save that clears the
margin on the anchor scale but is **clearly** the weaker net one-on-one.

- New helper `loses_head_to_head(w, l, d, threshold)`: true only when the
  challenger's score rate is strictly **below** `1 - threshold` (i.e. `< 0.45`).
  It is deliberately **not** the negation of `beats_best`: a level, winning,
  missing (`None`), or wedged (no-games / all-draws) h2h is **not** a clear loss
  and does not veto.
- `Best_tracker.consider`: promotes when `elo > best + best_margin` **and** the
  h2h, if present, is not a clear loss. A clear margin win with no h2h now
  promotes (the old code silently declined when `head_to_head_record is None`).

Rollback accounting and the first-check rule are unchanged.

---

## Bug B — epoch saves went to the shared `checkpoints/`, not the run dir

### Symptom

`ppo_epoch_<N>.pt` saves landed in the shared `bobble_model/checkpoints/` while
`best_p14.pt` correctly went to `bobble_model/p14_run/checkpoints/`. Two runs
sharing the repo overwrite each other's epoch saves.

### Root cause

In `bobble_model/training_loop.py` the periodic full-checkpoint save hardcoded the
path:

```python
os.makedirs('checkpoints', exist_ok=True)
save_checkpoint(f'checkpoints/ppo_epoch_{epoch}.pt', ...)
```

`--best-path` steered `best.pt` into the run dir, but the epoch saves ignored it.

### Fix

`bobble_model/training_loop.py`: epoch saves now go under an `epoch_ckpt_dir`
derived from `best_path` (its directory), overridable by a new `--checkpoint-dir`
/ `checkpoint_dir=` argument. With the default `--best-path checkpoints/best.pt`
the dir resolves to `checkpoints/`, so default behaviour is unchanged.

```python
epoch_ckpt_dir = checkpoint_dir if checkpoint_dir else os.path.dirname(best_path)
if not epoch_ckpt_dir:
    epoch_ckpt_dir = 'checkpoints'
...
os.makedirs(epoch_ckpt_dir, exist_ok=True)
save_checkpoint(os.path.join(epoch_ckpt_dir, f'ppo_epoch_{epoch}.pt'), ...)
```

---

## Tests (no Godot)

`tests/unit/test_phase10.py` (Bug A, T1 + nearby cases):

- `test_loses_head_to_head_only_on_a_clear_loss` — the new helper: `< 0.45` loses;
  level / near-level / win / no-games / all-draws do **not**.
- `test_clear_h2h_loss_blocks_even_past_the_margin` — a margin winner that clearly
  loses the h2h (`4-16-0`, 0.20) is **not** promoted.
- `test_level_h2h_does_not_veto_a_clear_margin_winner` — the core fix: `961.5 ->
  1203.1` with a level h2h (`10-10-0`) now promotes.
- `test_margin_win_at_the_boundary_does_not_promote` — exactly `+25` does not,
  `+25.1` does.
- `test_margin_win_with_no_h2h_record_still_promotes` — a clear margin win with
  `h2h=None` promotes (resume / first-challenge case).
- `test_long_run_2_sequence_now_promotes` — replays the exact logged Elo sequence;
  best no longer freezes at 961.5 (ends at 1203.1 / epoch 100).
- Existing `test_higher_elo_and_wins_h2h_becomes_best`,
  `test_noise_margin_blocks_a_save_only_slightly_ahead`,
  `test_margin_can_be_tuned_off`, rollback tests — still pass.

`tests/mock/test_train_short.py` (Bug B, T2, real `ppo_train` on mock games):

- `test_epoch_saves_land_in_run_dir_not_shared` — epoch saves follow `--best-path`
  into `p14_run/checkpoints/`, and the shared `checkpoints/` is not written.
- `test_checkpoint_dir_overrides_best_path_dir` — explicit `--checkpoint-dir` wins.
- `test_default_best_path_keeps_shared_checkpoints_dir` — default path unchanged.

Full non-live suite: **203 passed** (`python -m pytest -m "not live" -q`), up from
195 (8 new tests). No C# was touched, so no `dotnet build` was needed.

## Smoke (real Godot, T4)

A standalone harness drove the **real** live-Elo best-gate path against one real
Godot on shm index 11 (clear of the sibling eval on 0–5/10, ~170 MB, RAM kept
>1.5 GB free). It scored two real nets against the chaser/defender anchors, seeded
the lower scorer as the current best, and let the higher scorer challenge through
the fixed gate with a **real** head-to-head:

```
random_init anchor-Elo 993.9  |  best_p11_ln anchor-Elo 1076.6
[best seed] epoch 0 anchor-Elo 993.9 -> first best at epoch 0, elo 993.9
[challenger best_p11_ln] epoch 10 anchor-Elo 1076.6 h2h (13, 2, 1)
  -> new best at epoch 10: elo 1076.6 > 993.9+25 and h2h W13 L2 D1 not a clear loss
PROMOTED after margin-clearing check: True
best_smoke.pt now at epoch: 10
best_smoke.pt in RUN dir   : True
ppo_epoch_0.pt in RUN dir  : True
NOT in shared dir          : True
SMOKE_PASS=True
```

- **Bug A:** the challenger cleared the best by **+82.6 Elo** (past the 25 margin),
  won a real 16-game h2h **13-2-1**, and the gate **promoted** `best` from epoch 0
  to epoch 10 — the exact move that never happened in long run 2.
- **Bug B:** both `best_smoke.pt` and `ppo_epoch_0.pt` landed in the run's own
  `checkpoints/` dir, with nothing written to the shared `bobble_model/checkpoints`.

The smoke's scratch run dir, its spare-Godot shm file, and the diagnostic script
were removed after the run; no `.pt` files were moved, deleted, or overwritten.

## Files changed

- `bobble_model/live_elo.py` — margin-authoritative gate + `loses_head_to_head`.
- `bobble_model/training_loop.py` — `checkpoint_dir` / `--checkpoint-dir`, epoch
  saves under the run dir.
- `tests/unit/test_phase10.py`, `tests/mock/test_train_short.py` — regression tests.

Fix commit: **`eb82e5a`**.
