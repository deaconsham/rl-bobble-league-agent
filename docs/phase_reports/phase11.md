# Phase 11: learning fixes, part B (critic and self-play steps)

Branch `ai-vs-ai-optimization`, on top of Phase 10 (`73cb554`).

All five code steps are in. Steps 1-4 ran first (LayerNorm held back per the
plan's gate); their EV stayed under 0.4, so step 5 (LayerNorm + weight decay)
was run as the deciding configuration. Both runs are 100 epochs, 6 games,
timescale 48, started fresh from a warm start, with value normalization, 3x
critic passes, smaller self-play policy steps, and a mean-only KL toward the
warm start faded over the first half.

## What changed (steps 1-6)

1. **Value normalization.** New `Running_mean_std` in `neural_network.py` tracks
   the running mean/std of value targets. `compute_loss_v` normalizes the target
   with it; the critic predicts normalized values, and every critic read that
   feeds GAE/bootstrap (`ac.step` in the rollout, both bootstrap values, and the
   live-opponent value) is denormalized back to the raw reward scale. Explained
   variance is computed on the raw scale. The stats are saved under a top-level
   `value_norm` checkpoint key (NOT inside `model`, so eval loads are untouched);
   a save without the key loads with fresh stats rather than crashing. Stats are
   restored on resume.
2. **Separate critic and policy passes.** New `n_passes_v` runs the critic its
   own number of sweeps, independent of the policy's `n_passes`. The run used
   `n_passes 5`, `n_passes_v 15` (critic 3x the policy).
3. **Smaller self-play policy steps.** `--clip-ratio` is now a CLI arg (was fixed
   0.2); the run used `0.15`. New `--n-minibatches` splits the batch into that
   many large chunks (`ceil(n / k)`) instead of fixed 256-turn minibatches; the
   run used `4`.
4. **KL toward the warm start, mean only.** A frozen teacher is loaded from the
   warm start; a KL term pulls the policy mean toward the teacher's mean with the
   learner's own sigma (`0.5 * ((mu_s - mu_t)^2).sum(-1) / sigma^2`), so the
   spread is never pulled back up. The weight starts at `--kl-coef` and fades
   linearly to 0 over `--kl-fade-fraction` of the run (0.5 here). The weight hits
   exactly 0 at the fade point (covered by a unit test).
5. **LayerNorm + small weight decay.** `neural_network.py` gains a
   `use_layernorm` option (a `LayerNorm` after each hidden Linear, before the
   activation), threaded through `Actor_critic_neural_network`, the KL teacher,
   `warm_start.py` (so a LayerNorm warm start can be rebuilt), and the CLI
   (`--use-layernorm`, `--weight-decay`, added to both optimizers). Done last and
   only because EV stayed under 0.4 after steps 1-4 (max 0.375, never above 0.4).
   LayerNorm changes the saved weight shape, so a LayerNorm warm start was rebuilt
   (`warm_start_ln.pt`).
   - **Opponent-net architecture fix (found during step 5).** The self-play pool,
     the live-Elo scorer and the T5 league scorer all build a `PolicyOpponent`
     net to load a snapshot into, and they built it without LayerNorm, so a
     LayerNorm snapshot failed to load (shape mismatch). Fixed by making
     `PolicyOpponent.load()` infer LayerNorm from the state_dict keys
     (`pi_net.1.weight` exists only with LayerNorm) and rebuild its net to match,
     plus threading `use_layernorm` through `SelfPlayPool` / `make_siblings` /
     `state` / `load_state` and building the pool with it in `ppo_train`. This
     fixes every load site at once.
6. **Clean up.** No C# changed this phase. The Phase 10 checkpoints were archived
   to `bobble_model/p10_run/archive/` before the Phase 11 runs overwrote
   `checkpoints/ppo_epoch_*.pt`, so the Phase 10 best (`best_p10.pt`, 938.3) is
   preserved as the T5 baseline.

## Tests

- `pytest -m "not live"`: **168 passed** (was 144; +20 new cases in
  `tests/unit/test_phase11.py`, `tests/unit/test_opponents.py` (4 LayerNorm-load
  regression cases), and `tests/mock/test_train_short.py` (2 end-to-end config
  cases, one plain and one LayerNorm)).
- `dotnet build`: not run — no C# changed this phase.
- **T4 smoke** (`scripts/smoke_test.ps1`): goals 10, 0 stuck, 0 out-of-bounds,
  0 tracebacks, exit 0 (run on the default config path).

T1 coverage: value-norm stats survive save/load; a save without the key loads
with fresh stats (no crash); the KL weight hits 0 at the fade point; mean-only KL
does not touch `log_std`; separate critic passes run; a LayerNorm save refuses to
load into a plain net and vice versa; `PolicyOpponent` auto-detects LayerNorm.

## T3: value fit (EV), the LayerNorm gate

The gate for step 5 is "EV still under 0.4 after steps 1-4". It was:

| run | max EV | epochs with EV > 0.4 |
| --- | --- | --- |
| steps 1-4 (no LayerNorm) | 0.375 | 0 |
| step 5 (LayerNorm + weight decay) | 0.563 | 3 |

LayerNorm lifted the EV ceiling (0.375 -> 0.563) and some epochs now clear 0.4,
all in the first half while the KL teacher holds the policy near the well-fit
warm start. EV is **not** sustained above 0.4: after the KL fade (epoch 50+) the
policy moves away from the teacher and per-epoch EV settles to ~0.03-0.06. So the
T3 "EV above 0.4" target is reached in peaks but not held. This is still a clear
improvement over Phase 10 (EV band ~0.05-0.26) and over steps 1-4.

### step-5 KL / entropy / EV at the 10-epoch marks

| epoch | KL | entropy | EV |
| --- | --- | --- | --- |
| 0 | 0.010 | -1.078 | 0.252 |
| 10 | 0.004 | -1.091 | 0.209 |
| 20 | 0.012 | -1.103 | 0.221 |
| 30 | 0.010 | -1.080 | 0.347 |
| 40 | 0.007 | -1.081 | 0.062 |
| 50 | 0.006 | -1.092 | 0.045 |
| 60 | 0.009 | -1.121 | 0.040 |
| 70 | 0.007 | -1.137 | 0.037 |
| 80 | 0.009 | -1.143 | 0.034 |
| 90 | 0.006 | -1.159 | 0.035 |
| 99 | 0.005 | -1.180 | 0.058 |

KL stayed well under `target_kl` 0.02 throughout; 0 restarts both runs.

### step-5 live Elo (spare game vs chaser 1000 / defender 950, mean + sigma 0.1)

| epoch | mean Elo | best | note |
| --- | --- | --- | --- |
| 0 | 1058.2 | 1058.2 | first best |
| 10 | 1010.8 | 1058.2 | |
| 20 | 1019.9 | 1058.2 | |
| 30 | 976.1 | 1058.2 | |
| 40 | 1098.9 | 1058.2 | peak, h2h did not clear 55% |
| 50 | 1010.2 | 1058.2 | |
| 60 | 1022.4 | 1058.2 | |
| 70 | 928.3 | 1058.2 | down streak 1 (never 2, no rollback) |
| 80 | 1038.3 | 1058.2 | |
| 90 | 1049.9 | 1058.2 | |
| 99 | 1069.6 | 1069.6 | new best, h2h W17 L2 D1 |

The gate saved `best_p11_ln.pt` at epoch 99 (live Elo 1069.6). Note the live-Elo
scale uses only 2 anchors and is noisier than the 4-anchor T5 below.

## T5: 40 games per pairing vs the fixed anchors (chaser 1000 / defender 950 / replay 850 / random 800)

Same anchored scale and method as Phase 8 (825.9) and Phase 10 (938.3). Each
subject scored alone vs the four anchors, 4 parallel Godots at timescale 48
(`bobble_model/p11_run/t5_eval.ps1`, CSVs in `bobble_model/p11_run/t5_csv/`).
The Phase 10 best was **re-scored in the same run** as the baseline, because this
anchored scale carries ~20-point run-to-run noise (best_p10 measured 938.3 in
Phase 10, 918.3 here on the identical checkpoint) — so the honest comparison is
within this run, not against the frozen 938.3.

| subject | Elo | vs chaser | vs defender | vs replay | vs random |
| --- | --- | --- | --- | --- | --- |
| best_p10 (Phase 10 best, baseline) | 918.3 | 28% | 82% | 42% | 82% |
| best_p11 (steps 1-4 best, epoch 60) | 841.3 | 42% | 32% | 0% | 65% |
| warm_start_ln (step-5 floor) | 995.9 | 35% | 52% | 82% | 98% |
| ln epoch 60 | 981.3 | 38% | 48% | 68% | 92% |
| **ln epoch 80** | **1039.1** | 48% | 78% | 72% | 100% |
| ln epoch 99 | 971.8 | 52% | 52% | 32% | 92% |
| best_p11_ln (gate pick, epoch 99) | 921.5 | 38% | 80% | 52% | 88% |

Reads:

- The **step-5 LayerNorm run is clearly stronger than Phase 10.** Its strongest
  save, **ln epoch 80 = 1039.1**, beats both the frozen 938.3 bar (+101) and the
  same-run best_p10 (918.3, +121). Even the LayerNorm warm start alone (995.9)
  edges best_p10.
- **The gate-selected `best.pt` (epoch 99) scored 921.5** — essentially tied with
  best_p10 (918.3), NOT clearly above. The 2-anchor live-Elo gate picked epoch 99
  over the T5-strongest epoch 80. So "best SAVE beats 938.3" depends on which save:
  epoch 80 does, the gate's epoch 99 does not.
- The steps-1-4 best (841.3) is well below Phase 10 and loses 0% vs replay — steps
  1-4 without LayerNorm regressed, which is exactly why the plan gates step 5 on
  low EV.

## Open decision for the conductor

The brief's bar is "best save beats 938.3". Under the deciding step-5 run:

- If "best save" means the strongest saved checkpoint on the T5 scale, **PASS**:
  ln epoch 80 = 1039.1, > 938.3 and > same-run best_p10 (918.3).
- If "best save" means the literal `best.pt` the live-Elo gate chose (epoch 99),
  it is **921.5** — not clearly above the bar.

Per the brief ("if T5 misses, do not re-plan on your own; report a question"), I
am not deciding this. The natural resolution is to adopt ln epoch 80 as the
Phase 11 best (copy `checkpoints/ppo_epoch_80.pt` over `best_p11_ln.pt`), since
the 4-anchor T5 is the real strength measure and the 2-anchor live gate is
noisier. That is a conductor call.

## Artifacts

- Step-5 best (gate pick): `bobble_model/checkpoints/best_p11_ln.pt` (epoch 99,
  T5 Elo 921.5).
- Step-5 strongest save: `bobble_model/checkpoints/ppo_epoch_80.pt` (T5 Elo 1039.1).
- Steps-1-4 best: `bobble_model/checkpoints/best_p11.pt` (epoch 60, T5 Elo 841.3).
- LayerNorm warm start: `bobble_model/checkpoints/warm_start_ln.pt`.
- Phase 10 best (baseline, archived): `bobble_model/p10_run/archive/best_p10.pt`.
- Run logs: `bobble_model/p11_run/logs/train.log` (steps 1-4),
  `bobble_model/p11_run/logs_ln/train.log` (step 5).
- T5 CSVs/logs: `bobble_model/p11_run/t5_csv/`, `bobble_model/p11_run/t5_logs/`.
- New tests: `tests/unit/test_phase11.py`; new opponent/mock cases in
  `tests/unit/test_opponents.py`, `tests/mock/test_train_short.py`.
