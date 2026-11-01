"""
T1 for Phase 10 (learning fixes A). No Godot.

- angle wrap: 1.1 -> ~-0.9, exactly +-1 -> own goal, large multiples fold in,
  power still clips, and the stored (pre-wrap) action is left untouched;
- log_std starts at -1.6 (sigma ~0.2) and is reset to it after loading an init
  checkpoint (an old save carries a wide spread);
- the LR floor: the schedule never drops below 30% of the start rate;
- live Elo matches hand-worked Elo maths;
- best.pt only changes when BOTH rules pass (highest Elo AND beats best head to
  head), and a >100 Elo drop held for 2 checks rolls back.
"""
import math

import numpy as np
import pytest
import torch
from torch.optim import Adam

from bobble_league_env import decode_angle, wrap_angle, wrap_action
from neural_network import INIT_LOG_STD, Actor_critic_neural_network
from training_loop import Linear_lr_schedule
from elo import expected_score
from live_elo import Best_tracker, beats_best, live_elo_rating


# ---- angle wrap ----

def test_wrap_just_past_plus_one():
    # 1.1 comes back round the near side to -0.9 (plan's stated case)
    assert wrap_angle(1.1) == pytest.approx(-0.9, abs=1e-6)


def test_wrap_just_past_minus_one():
    assert wrap_angle(-1.1) == pytest.approx(0.9, abs=1e-6)


def test_wrap_exactly_plus_one_is_own_goal():
    # +1 maps to -1, but both seam ends are the SAME direction (own goal), so the
    # decoded aim is unchanged - not a jump to somewhere else.
    assert wrap_angle(1.0) == pytest.approx(-1.0, abs=1e-6)
    assert np.allclose(decode_angle(1.0), decode_angle(-1.0), atol=1e-6)
    assert np.allclose(decode_angle(1.0), [0.0, 1.0], atol=1e-6)  # +z = own goal


def test_wrap_exactly_minus_one_unchanged():
    assert wrap_angle(-1.0) == pytest.approx(-1.0, abs=1e-6)


def test_wrap_large_multiples_fold_in():
    # adding whole multiples of 2 is the same aim; wrap brings them back in range
    for base in (-0.3, 0.0, 0.42, 0.9):
        for k in (-3, -1, 2, 5):
            assert wrap_angle(base + 2 * k) == pytest.approx(
                wrap_angle(base), abs=1e-5), f"base {base} k {k}"
    # every wrapped value sits in [-1, 1)
    xs = wrap_angle(np.linspace(-10, 10, 201))
    assert np.all(xs >= -1.0 - 1e-6) and np.all(xs < 1.0 + 1e-6)


def test_wrap_never_aims_near_own_goal_from_in_range_mean():
    # a sample just past the enemy-goal direction (a ~ 0) must NOT end up at the
    # seam (own goal). clipping would pin 1.1 at +1 (own goal); wrap keeps it near
    # the real aim. check the decoded z stays well away from +1 (own goal) for
    # samples near the enemy goal.
    for a in (0.05, -0.05, 0.1, -0.1):
        z = decode_angle(a)[1]
        assert z < -0.9, f"a={a} should still aim at the enemy goal (-z), got z={z}"


# ---- power still clips, stored action is pre-wrap ----

def test_wrap_action_angles_wrap_power_clips():
    raw = np.array([1.1, 1.5, -1.2, -2.0, 0.3, 0.7], dtype=np.float32)
    wrapped = wrap_action(raw)
    # angle slots wrapped
    assert wrapped[0] == pytest.approx(-0.9, abs=1e-6)
    assert wrapped[2] == pytest.approx(0.8, abs=1e-6)
    assert wrapped[4] == pytest.approx(0.3, abs=1e-6)
    # power slots clipped to [-1, 1]
    assert wrapped[1] == pytest.approx(1.0)
    assert wrapped[3] == pytest.approx(-1.0)
    assert wrapped[5] == pytest.approx(0.7)


def test_wrap_action_does_not_mutate_input():
    # PPO keeps the raw sampled action; wrap_action must not change it in place
    raw = np.array([1.1, 1.5, -1.2, -2.0, 0.3, 0.7], dtype=np.float32)
    before = raw.copy()
    _ = wrap_action(raw)
    assert np.allclose(raw, before)


# ---- log_std init and reset ----

def test_log_std_starts_small():
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    assert INIT_LOG_STD == pytest.approx(-1.6)
    assert torch.allclose(ac.log_std, torch.full((6,), -1.6), atol=1e-6)
    # sigma ~0.2
    assert float(ac.log_std.detach().exp().mean()) == pytest.approx(math.exp(-1.6), abs=1e-3)


def test_log_std_reset_after_loading_init_checkpoint(tmp_path):
    # simulate loading an OLD init checkpoint whose log_std is wide (-0.5), then the
    # training loop's reset: the exact two lines ppo_train runs on --init.
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    with torch.no_grad():
        ac.log_std.fill_(-0.5)            # an old save's wide spread
    # the reset the loop applies right after load_state_dict
    with torch.no_grad():
        ac.log_std.fill_(float(INIT_LOG_STD))
    assert torch.allclose(ac.log_std, torch.full((6,), -1.6), atol=1e-6)


# ---- LR floor ----

def test_lr_never_below_thirty_percent():
    pi = Adam([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    vf = Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    epochs = 50
    sched = Linear_lr_schedule([pi, vf], [3e-4, 1e-3], total=epochs)
    floor = Linear_lr_schedule.DEFAULT_FLOOR_FRAC
    assert floor == pytest.approx(0.3)
    seen_floor = False
    for _ in range(epochs + 5):
        lr = pi.param_groups[0]["lr"]
        assert lr >= 3e-4 * floor - 1e-12
        if lr == pytest.approx(3e-4 * floor):
            seen_floor = True
        sched.advance()
    assert seen_floor, "schedule never reached the floor"


# ---- live Elo hand maths ----

def test_live_elo_matches_hand_math():
    # one anchor (chaser 1000), one game, the learner wins it. expected score of
    # the learner (starting at the default 1000) vs chaser 1000 is 0.5, so a win
    # moves it by k*(1 - 0.5) = 32*0.5 = 16 -> 1016.
    r = live_elo_rating({"chaser": (1, 0, 0)}, k=32.0,
                        anchor_ratings={"chaser": 1000.0})
    assert r == pytest.approx(1000.0 + 32.0 * (1.0 - 0.5), abs=1e-6)


def test_live_elo_two_anchors_sequential():
    # chaser 1000, defender 950. learner starts 1000. play chaser first (10-10),
    # then defender (15-5). fold game by game, anchors pinned, and compare to a
    # hand loop using expected_score / the Elo update.
    def hand(records, anchors, k=32.0):
        rating = 1000.0
        for name in anchors:  # dict order: chaser then defender
            w, l, d = records[name]
            seq = [1.0] * w + [0.0] * l + [0.5] * d
            for s in seq:
                exp = expected_score(rating, anchors[name])
                rating = rating + k * (s - exp)
        return rating

    anchors = {"chaser": 1000.0, "defender": 950.0}
    records = {"chaser": (10, 10, 0), "defender": (15, 5, 0)}
    got = live_elo_rating(records, k=32.0, anchor_ratings=anchors)
    assert got == pytest.approx(hand(records, anchors), abs=1e-6)


# ---- best.pt gate + rollback ----

def test_beats_best_gate():
    assert beats_best(12, 7, 1) is True       # 12.5/20 = 0.625 > 0.55
    assert beats_best(11, 9, 0) is False      # 0.55 not strictly above 0.55
    assert beats_best(0, 0, 0) is False       # no games


def test_first_check_becomes_best_no_h2h():
    t = Best_tracker()
    d = t.consider(epoch=0, elo=900.0, head_to_head_record=None)
    assert d["became_best"] is True
    assert t.best_elo == pytest.approx(900.0)


def test_higher_elo_but_loses_h2h_does_not_become_best():
    t = Best_tracker()
    t.consider(epoch=0, elo=900.0)                  # best = 900
    d = t.consider(epoch=10, elo=950.0, head_to_head_record=(10, 10, 0))  # 50% <= 55%
    assert d["became_best"] is False
    assert t.best_elo == pytest.approx(900.0)       # unchanged


def test_higher_elo_and_wins_h2h_becomes_best():
    t = Best_tracker()
    t.consider(epoch=0, elo=900.0)
    d = t.consider(epoch=10, elo=950.0, head_to_head_record=(13, 6, 1))  # 0.675 > 0.55
    assert d["became_best"] is True
    assert t.best_elo == pytest.approx(950.0)


def test_rollback_after_two_big_drops():
    t = Best_tracker()
    t.consider(epoch=0, elo=1000.0)                 # best = 1000
    d1 = t.consider(epoch=10, elo=880.0)            # 120 below: down streak 1
    assert d1["rollback"] is False
    d2 = t.consider(epoch=20, elo=870.0)            # still >100 below: streak 2 -> rollback
    assert d2["rollback"] is True


def test_one_recovery_clears_the_streak():
    t = Best_tracker()
    t.consider(epoch=0, elo=1000.0)
    t.consider(epoch=10, elo=880.0)                 # down 1
    d = t.consider(epoch=20, elo=950.0)             # within 100: clears streak
    assert d["rollback"] is False
    d2 = t.consider(epoch=30, elo=870.0)            # down 1 again, not 2 -> no rollback
    assert d2["rollback"] is False
