# Phase 7 report: the overnight run

Date: 2026-11-01. Branch: `ai-vs-ai-optimization`. Written in Phase 8.

Phase 7 was the first long self-play run on the Phase 0-6 code. Two runs were
done: a warm-started run (`p7_run`) and a from-scratch run (`p7_scratch`). This
report records both, their training health, their strength (Elo and win rates),
how they compare at matched epochs, why the warm run stopped at epoch 251, and
whether the model is still easy to beat. The answer to the last one is yes.

## The two runs

| Run | Folder | Epochs | Start | Notes |
|---|---|---|---|---|
| Warm start | `bobble_model/p7_run` | 0 -> 251 of 300 planned | `warm_start.pt` (cloned from the chaser) | 4 games, self-play. Stopped on purpose at epoch 251, not a crash. |
| From scratch | `bobble_model/p7_scratch` | 0 -> 39 of 40 planned | `random_init.pt` | 4 games, live self-play. Ran to its planned end. |

## Training health

Numbers read from `p7_run/warm_train.log` and `p7_scratch/scratch_train.log`.

| Sign | Warm start (epoch 0) | Warm start (epoch 251) | Scratch (epoch 0) | Scratch (epoch 39) | Healthy range |
|---|---|---|---|---|---|
| Entropy | 5.53 | 4.98 | 5.52 | 5.58 | falling slowly, not to 0 |
| Explained variance | 0.08 | 0.25-0.33 | 0.09 | 0.07 | above 0 and rising |
| KL | 0.013 | 0.009-0.011 | 0.011 | 0.0006 | mostly under 0.02 |
| Clip fraction | 0.19 | 0.11 | 0.13 | 0.0002 | 0.1-0.3 |
| Learning rate | 3.0e-04 | 4.9e-05 | 3.0e-04 | 7.5e-06 | above the floor |
| Turn length (mean / max s) | 3.69 / 7.95 | 3.63 / 7.40 | 3.29 / 7.05 | 3.29 / 6.93 | settles under the limit |
| Restarts | 0 | 0 | 0 | 0 | rare and logged |

What the numbers say:

- The warm run's critic improved (EV 0.08 -> ~0.33) but the policy barely moved:
  entropy only fell 5.53 -> 4.98, so the aim stayed almost as random as it
  started. This is the Phase 10 target (small starting spread, wrapped angle).
- The scratch run **stopped learning** well before its end: by epoch 39 the KL
  was 0.0006, the clip fraction 0.0002 and the learning rate had decayed to
  7.5e-06. It was no longer taking meaningful steps. This is the Phase 10
  learning-rate-floor target.
- Neither run crashed or hung: 0 restarts in both.

## Strength (Elo)

Elo is anchored to the fixed held-out bots: chaser 1000, defender 950,
replay 850, random 800. Section 1 of the final eval
(`p7_run/eval_results.json`, 20 games a pairing, sparse ladder 0/50/100/150/200/250):

| Model | Elo |
|---|---|
| chaser (anchor) | 1000.0 |
| defender (anchor) | 950.0 |
| replay (anchor) | 850.0 |
| ppo_epoch_150 | 829.3 |
| ppo_epoch_50 | 821.4 |
| ppo_epoch_250 | 807.3 |
| random (anchor) | 800.0 |
| ppo_epoch_0 | 645.0 |
| ppo_epoch_200 | 621.3 |
| ppo_epoch_100 | 588.8 |

Every trained save lands below the chaser (1000), and most sit at or below the
random bot (800). The best save (epoch 150) is 829, barely above random. Elo did
not rise across the run; it bounced (50 and 150 up, 100/200 down), which matches
the flat entropy: the model was not steadily getting better.

Final warm checkpoint (epoch 250) against the test-only bots, 25 games each:
defender 36% (W9 L14 T2), replay 24% (W6 L12 T7). The replay "winning move"
still scored 80% of the time against epoch 250 - the one fixed counter-move the
model never learned to stop.

The epoch-60 save was re-tested over 40 games in Phase 8 and placed at **Elo
825.9** (vs chaser 32%, defender 60%, replay 42%, random 90%); that number is
the bar for later phases. See `docs/phase_reports/phase8.md`.

## Warm vs scratch at matched epochs

From `eval_results.json`, 20 games a pairing (win rate, learner point of view):

| Epoch | Opponent | Warm | Scratch |
|---|---|---|---|
| 30 | chaser | 50% | 90% |
| 30 | defender | 40% | 30% |
| 30 | replay | 0% | 0% |
| 40 | chaser | 45% | 45% |
| 40 | defender | 35% | 50% |
| 40 | replay | 0% | 0% |

The warm start did not give the lasting edge hoped for. It was ahead only very
early (and even then the scratch run beat the chaser more often at epoch 30,
because it trained against a live opponent rather than self-play). By epoch 40
the two are even. Both lose every replay game. The scratch run also times out a
lot against defender/replay, which shows it had not learned to finish.

## Why the warm run stopped at 251

It was stopped on purpose, not by a crash and not by `--epochs`. The Phase 7
task chat ran `p7_run/stop_warm.ps1` at 18:26:46, which found the warm run's
`training_loop.py` process (log-dir `runs\warm`) and killed it, then stopped its
four Godot games. The chat believed 250 was the target, so it cut the run one
epoch after the epoch-250 checkpoint was written (the loop had already started
epoch 251 when it was killed, so epoch 251 has a log block but no saved
checkpoint). The run was healthy at the time - this was an operator decision,
not a failure.

This is exactly the problem Phase 8 step 4 fixes: a long run's stop epoch must
come only from `--epochs`, never from a helper script. See `phase8.md`.

## Is it still easy to beat?

Yes.

- Best Elo across the run (epoch 150) is 829 - barely above the random bot (800)
  and far below the chaser (1000).
- The epoch-60 re-test confirms it: 32% against the chaser over 40 games.
- The replay bot's single fixed move scored 80% against the final model.
- The model wins easily only against `random` (85-90%).

The likely cause is the aim: entropy stayed near its start the whole run, so the
policy shoots almost as randomly as it began. Phase 10 (small starting spread,
wrapped angle, learning-rate floor, live Elo and `best.pt`) targets this directly.
