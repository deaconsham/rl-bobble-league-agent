# Phase 14 eval: long run 2 saves vs Phase 11 best

Branch `ai-vs-ai-optimization`. Evaluates the "long run 2" training run (200
epochs, 0-199) that started from `best_p11_ln.pt` (Phase 11 best, LN epoch 80)
and finished at 15:36 on 2026-11-02.

**Outcome: PASS. The long-run-2 policy at epoch 100 (`ppo_epoch_100.pt`, Elo
1156.1) beats the Phase 11 best `best_p11_ln.pt` (1103.5) by +52.6 Elo in the
same within-run 40-game T5 — well beyond the ~20-point run-to-run noise — and
wins the head-to-head 27-13.** The policy peaks around epoch 100 and then drifts
downward (epoch 199 is back to 1074.2, barely above the Phase 11 best), so the
gain is real but came mid-run, not at the end. The best save is **epoch 100**,
not the final epoch-199 save and not `best_p14.pt`.

## Method

One within-run `league_eval.py` run so every rating is on the same anchored
scale (Phase 11 found the anchored scale drifts ~20 pts run to run, so only
within-run comparisons are trusted). 40 games per pairing, 4 fresh Godot games
(timescale 48) on shm indices 0-3, parallel pairings. Anchors pinned: random
800, replay 850, defender 950, chaser 1000.

Players: the Phase 11 best as the bar, plus long-run-2 saves at epochs 50, 100,
130, 150, 170, 199, plus `best_p14.pt` (which differs from epoch 0, so it was
included). All eight checkpoints loaded eligibly (action format >= 3).

## T5 — within-run Elo (one 40-game league run)

| Model | Elo | notes |
| --- | --- | --- |
| **ppo_epoch_100.pt (Phase 14 best)** | **1156.1** | long-run-2 peak |
| best_p11_ln.pt (Phase 11, the bar) | 1103.5 | |
| ppo_epoch_170.pt | 1092.4 | |
| ppo_epoch_199.pt | 1074.2 | final save; drifted down from the peak |
| ppo_epoch_130.pt | 1021.5 | |
| ppo_epoch_150.pt | 1005.3 | |
| chaser (anchor) | 1000.0 | |
| ppo_epoch_50.pt | 993.1 | mid-training, below chaser |
| defender (anchor) | 950.0 | |
| best_p14.pt | 947.8 | never updated past the epoch-0 region (see bug) |
| replay (anchor) | 850.0 | |
| random (anchor) | 800.0 | |

The within-run live-Elo trace agrees on the shape: mean live Elo peaked 1203.1
@ epoch 100, 1102.2 @ 130, 1025.5 @ 199 (`train.log`).

## Does a Phase 14 save beat the bar?

**Yes, clearly.** `ppo_epoch_100.pt` at 1156.1 beats `best_p11_ln.pt` at 1103.5
by **+52.6 Elo**, far outside the ~20-pt noise band. Head-to-head the two played
40 games and epoch 100 won **27-13-0** (67.5% score rate) — a decisive margin,
not a coin flip.

For reference, the other candidates head-to-head vs `best_p11_ln.pt` (as
best_p11_ln's W-L-D, so a low W means the p14 save won):

| p14 save | best_p11_ln W-L-D vs it | winner |
| --- | --- | --- |
| ppo_epoch_100.pt | 13-27-0 | **epoch 100 (+14)** |
| ppo_epoch_199.pt | 12-27-1 | epoch 199 |
| ppo_epoch_150.pt | 15-25-0 | epoch 150 |
| ppo_epoch_130.pt | 22-18-0 | best_p11_ln |
| ppo_epoch_170.pt | 30-10-0 | best_p11_ln |
| ppo_epoch_50.pt | 30-9-1 | best_p11_ln |
| best_p14.pt | 20-18-2 | ~even |

Head-to-head wins and the Elo ranking disagree for epochs 170/199 (Elo ranks
170 > 199, but 199 wins their direct game 20-20 is even and 199 beats the bar
head-to-head while 170 loses it 10-30). This is the usual non-transitivity of a
sparse round-robin; the anchored Elo fold is the single ranking the acceptance
uses, and on it epoch 100 is unambiguously on top.

## Is the AI still easy to beat?

Not by the fixed anchors — the Phase 14 best beats every anchor — but **chaser
stays close**, which is the same ceiling Phase 11 hit. Anchor records for the
Phase 14 best (`ppo_epoch_100.pt`), score rate = (W + 0.5*D)/40:

| opponent | W-L-D | learner score rate |
| --- | --- | --- |
| random | 39-1-0 | 97.5% |
| replay | 38-1-1 | 96.3% |
| defender | 32-5-3 | 83.8% |
| chaser | 21-16-3 | 56.3% |

So epoch 100 does **not lose** to chaser or defender (it wins both series), but
chaser is only a 56% edge — the agent is still beatable by an aggressive
ball-chaser about 4 games in 10. The Phase 11 best was the same story (vs chaser
18-17-5, 51.3%), so Phase 14 improved the chaser matchup slightly but did not
break the ceiling. The final epoch-199 save regressed here to 18-17-5 (51.3% vs
chaser), matching its lower Elo.

## Repeated-move ("replay") exploit

The replay anchor replays a fixed "winning move"; the exploit rate is how often
that move still scores against the learner, i.e. `1 - learner_score_rate` vs
replay.

| policy (this run) | learner vs replay W-L-D | exploit rate |
| --- | --- | --- |
| **ppo_epoch_100.pt (Phase 14 best)** | 38-1-1 | **~3.8%** |
| best_p11_ln.pt (Phase 11 best) | 32-2-6 | ~12.5% |
| ppo_epoch_199.pt (final) | 34-5-1 | ~13.8% |

The exploit went **down** for the Phase 14 best: ~3.8% vs the Phase 11 best's
~12.5% in this same run, and vs Phase 12's 22%. So epoch 100 is both stronger on
Elo and more robust to the repeated-move exploit. The final epoch-199 save is
worse again (~14%), consistent with the general late-run drift.

(Note on baselines: Phase 12's report quoted Phase 11 at ~6% exploit measured in
a different run; re-scored here in the same run as Phase 14 it reads ~12.5%,
inside the known run-to-run swing. The within-run comparison — ep100 3.8% vs
best_p11_ln 12.5% — is the trustworthy one.)

## Epoch-save path issue (bug, not fixed here)

The run's periodic epoch saves were written to the **shared** checkpoints folder
`bobble_model\checkpoints\ppo_epoch_{0,10,...,190,199}.pt`, not to the run's own
`bobble_model\p14_run\checkpoints\`. Separately, `best_p14.pt` (in
`p14_run\checkpoints\`) was last written at 07:43 near epoch 0 and **never
updated past the epoch-0 region** — the run's live-Elo best-gate stayed at "best
961.5" (the epoch-0 reading) for all 200 epochs (`train.log`), so the true best
(epoch 100) was never promoted to `best_p14.pt`.

This eval worked around both by **copying** (not moving) the 21 long-run-2 epoch
saves (LastWriteTime 2026-11-02 after 07:30) into `p14_run\checkpoints\` and
scoring from the copies; sizes were verified identical. A sibling change on this
branch (commit `43bb437`) addresses the best-gate promotion and the run-dir save
location in the training code; this report only documents the symptom as seen in
long run 2 and did not touch training code.

## Files used

- Bar: `bobble_model\checkpoints\best_p11_ln.pt` (Phase 11 best, LN epoch 80).
- Scored (copies in `bobble_model\p14_run\checkpoints\`): `best_p14.pt`,
  `ppo_epoch_50.pt`, `ppo_epoch_100.pt`, `ppo_epoch_130.pt`, `ppo_epoch_150.pt`,
  `ppo_epoch_170.pt`, `ppo_epoch_199.pt`.
- Training log: `bobble_model\p14_run\logs\train.log` (200 epochs, 0 restarts,
  `train.done` = `TRAIN_DONE_0`).
- T5 driver: `bobble_model\p14_run\run_t5.ps1`; raw results
  `bobble_model\p14_run\t5_logs\t5.log`; Elo CSV
  `bobble_model\p14_run\t5_logs\elo.csv`.

## Verdict

**PASS.** Long run 2 produced a save (`ppo_epoch_100.pt`, Elo 1156.1) that beats
the Phase 11 best by +52.6 Elo within-run and 27-13 head-to-head, and lowers the
repeated-move exploit to ~3.8%. The gain peaks at epoch 100 and decays by epoch
199, so **epoch 100 is the Phase 14 best**, and if it is adopted as the new
champion it should be `ppo_epoch_100.pt`, not the final save or `best_p14.pt`.
The agent still only narrowly beats the chaser anchor (56%), so robustness vs an
aggressive chaser remains the open ceiling for a later phase.
