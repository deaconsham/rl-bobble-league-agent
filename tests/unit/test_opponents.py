"""
T1: the self-play opponent mix.

Checks that SelfPlayPool.reset() draws the chaser / live / frozen slots in roughly
the set proportions over 10,000 draws, that the live slot only appears once a live
net is attached, that sibling pools share one snapshot list, and that the chaser
still produces in-range angle+power actions.
"""
import numpy as np
import pytest

from neural_network import Actor_critic_neural_network
from opponents import (
    ChaserOpponent,
    DefenderOpponent,
    LiveOpponent,
    PolicyOpponent,
    RandomOpponent,
    ReplayOpponent,
    SelfPlayPool,
    TEST_ONLY_OPPONENTS,
    make_test_opponent,
    pfsp_weights,
)


def _fresh_net():
    net = Actor_critic_neural_network(obs_dim=14, act_dim=6, hidden=[32, 32])
    net.eval()
    return net


def _fill_snapshots(pool, n=3):
    net = _fresh_net()
    for _ in range(n):
        pool.add(net.state_dict())


def test_mix_matches_the_set_weights_over_10k_draws():
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=0)
    pool.attach_live(_fresh_net())
    _fill_snapshots(pool, 3)

    counts = {'chaser': 0, 'live': 0, 'self': 0}
    for _ in range(10000):
        pool.reset()
        counts[pool.name] += 1

    chaser = counts['chaser'] / 10000
    live = counts['live'] / 10000
    frozen = counts['self'] / 10000
    # targets: 0.15 / 0.30 / 0.55, allow +-0.03 sampling slack
    assert abs(chaser - 0.15) < 0.03, counts
    assert abs(live - 0.30) < 0.03, counts
    assert abs(frozen - 0.55) < 0.03, counts
    assert chaser + live + frozen == 1.0


def test_live_slot_absent_until_a_net_is_attached():
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=1)
    _fill_snapshots(pool, 3)  # snapshots exist but no live net yet

    names = set()
    for _ in range(2000):
        pool.reset()
        names.add(pool.name)
    # with no live net it is only chaser or a frozen copy, never "live"
    assert 'live' not in names
    assert names <= {'chaser', 'self'}


def test_is_live_current_tracks_the_live_slot():
    pool = SelfPlayPool(scripted_prob=0.0, live_prob=1.0, hidden=(32, 32), seed=2)
    pool.attach_live(_fresh_net())
    _fill_snapshots(pool, 2)
    pool.reset()
    # live_prob 1.0 with a live net -> always the live learner
    assert pool.name == 'live'
    assert pool.is_live_current() is True

    chaser_only = SelfPlayPool(scripted_prob=1.0, live_prob=0.0, hidden=(32, 32), seed=3)
    chaser_only.attach_live(_fresh_net())
    chaser_only.reset()
    assert chaser_only.name == 'chaser'
    assert chaser_only.is_live_current() is False


def test_siblings_share_one_snapshot_list_and_live_net():
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=4)
    pool.attach_live(_fresh_net())
    siblings = pool.make_siblings(6)
    assert len(siblings) == 6 and siblings[0] is pool

    # a snapshot added to any sibling is seen by all (shared list object)
    _fill_snapshots(siblings[2], 1)
    for sib in siblings:
        assert len(sib.snapshots) == 1
    _fill_snapshots(pool, 1)
    for sib in siblings:
        assert len(sib.snapshots) == 2

    # every sibling has a live slot, and each keeps its OWN current opponent
    for sib in siblings:
        assert sib.live is not None
    siblings[0].current = siblings[0].scripted
    siblings[1].current = siblings[1].live
    assert siblings[0].name == 'chaser'
    assert siblings[1].name == 'live'


def test_live_opponent_samples_and_records_its_turn():
    net = _fresh_net()
    live = LiveOpponent(net)
    raw = np.zeros(14, dtype=np.float32)
    a = live.act(raw)
    assert a.shape == (6,)
    assert live.last is not None
    assert live.last['obs'].shape == (14,)
    assert live.last['act'].shape == (6,)
    assert np.isfinite(live.last['val'])
    assert np.isfinite(live.last['logp'])


def test_chaser_output_shape_and_range():
    chaser = ChaserOpponent(seed=0)
    raw = np.array([1.0, -2.0, 3, 4, -3, -4, 0, 5, 2, -6, -2, -6, 0, -8], dtype=np.float32)
    action = chaser.act(raw)
    assert action.shape == (6,)
    angles = action[0::2]
    powers = action[1::2]
    assert np.all(angles >= -1.0) and np.all(angles <= 1.0)
    assert np.all(powers >= -1.0) and np.all(powers <= 1.0)


def test_mix_without_frozen_copies_uses_live_or_chaser():
    # before the first snapshot exists, draws are only chaser or live
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=5)
    pool.attach_live(_fresh_net())
    names = set()
    for _ in range(2000):
        pool.reset()
        names.add(pool.name)
    assert names <= {'chaser', 'live'}
    assert 'self' not in names  # no frozen copies yet


# ---- PFSP + held-out test opponents ----

def test_pfsp_weights_favour_the_losers():
    # win rates: beats copy0 always, 50/50 copy1, loses to copy2 always
    w = pfsp_weights([1.0, 0.5, 0.0], power=2.0)
    assert w.shape == (3,)
    assert w.sum() == pytest.approx(1.0)
    # (1-wr)^2 = 0, 0.25, 1 -> the copy it loses to is weighted most
    assert w[2] > w[1] > w[0]
    # the always-beaten copy is almost never picked (only the eps floor)
    assert w[0] < 1e-3


def test_pfsp_weights_empty_and_all_beaten():
    assert pfsp_weights([]).shape == (0,)
    # every copy beaten (winrate 1) -> (1-wr)^2 all ~0, falls back to near-uniform
    w = pfsp_weights([1.0, 1.0, 1.0])
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 1.0 / 3.0, atol=1e-3)


def test_record_result_moves_win_rate_by_ema():
    pool = SelfPlayPool(hidden=(32, 32), seed=0)
    _fill_snapshots(pool, 2)
    assert pool.win_rates == [0.5, 0.5]
    pool.record_result(0, learner_won=1.0, alpha=0.1)
    # 0.9*0.5 + 0.1*1.0 = 0.55
    assert pool.win_rates[0] == pytest.approx(0.55)
    pool.record_result(1, learner_won=0.0, alpha=0.1)
    assert pool.win_rates[1] == pytest.approx(0.45)
    # out-of-range / None index is a no-op
    pool.record_result(None, 1.0)
    pool.record_result(99, 1.0)
    assert pool.win_rates == pytest.approx([0.55, 0.45])


def test_pfsp_selection_biases_toward_the_hard_copy():
    pool = SelfPlayPool(scripted_prob=0.0, live_prob=0.0, hidden=(32, 32),
                        pfsp=True, seed=7)
    _fill_snapshots(pool, 3)
    # copy 2 is the one the learner loses to most
    pool.win_rates[:] = [0.9, 0.5, 0.1]
    counts = [0, 0, 0]
    for _ in range(5000):
        counts[pool.pick_snapshot_index()] += 1
    # the hardest copy (index 2) is drawn the most, the easiest (0) the least
    assert counts[2] > counts[1] > counts[0]


def test_uniform_selection_when_pfsp_off():
    pool = SelfPlayPool(scripted_prob=0.0, live_prob=0.0, hidden=(32, 32),
                        pfsp=False, seed=3)
    _fill_snapshots(pool, 3)
    pool.win_rates[:] = [0.9, 0.5, 0.1]
    counts = [0, 0, 0]
    for _ in range(6000):
        counts[pool.pick_snapshot_index()] += 1
    # uniform: each ~2000, within sampling slack, regardless of win rate
    for c in counts:
        assert abs(c - 2000) < 300, counts


def test_test_only_opponents_never_enter_the_pool():
    # the held-out set is a known fixed list, and SelfPlayPool only ever plays
    # its chaser / live / a frozen snapshot -> never a defender or replay.
    assert set(TEST_ONLY_OPPONENTS) == {'random', 'chaser', 'defender', 'replay'}
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32),
                        pfsp=True, seed=11)
    pool.attach_live(_fresh_net())
    _fill_snapshots(pool, 3)
    names = set()
    for _ in range(3000):
        pool.reset()
        names.add(pool.name)
    # only the three training slots ever appear
    assert names <= {'chaser', 'live', 'self'}
    assert 'defender' not in names and 'replay' not in names


def test_make_test_opponent_builds_each_held_out_one():
    for name in TEST_ONLY_OPPONENTS:
        opp = make_test_opponent(name, seed=0)
        assert opp.name == name
    with pytest.raises(ValueError):
        make_test_opponent('not-a-real-opponent')


def test_defender_output_shape_and_range():
    defender = DefenderOpponent(seed=0)
    raw = np.array([0, 2, 1, 5, -1, 6, 0, 7, -2, -6, 0, -8, 2, -7], dtype=np.float32)
    action = defender.act(raw)
    assert action.shape == (6,)
    assert np.all(action[0::2] >= -1.0) and np.all(action[0::2] <= 1.0)
    assert np.all(action[1::2] >= -1.0) and np.all(action[1::2] <= 1.0)


def test_replay_opponent_repeats_and_wraps():
    actions = [np.full(6, 0.1, dtype=np.float32), np.full(6, -0.2, dtype=np.float32)]
    replay = ReplayOpponent(actions)
    raw = np.zeros(14, dtype=np.float32)
    a0 = replay.act(raw)
    a1 = replay.act(raw)
    a2 = replay.act(raw)  # wraps back to actions[0]
    assert np.allclose(a0, 0.1)
    assert np.allclose(a1, -0.2)
    assert np.allclose(a2, 0.1)
    replay.reset()
    assert np.allclose(replay.act(raw), 0.1)


def test_default_replay_is_a_full_power_straight_shot():
    replay = ReplayOpponent()
    a = replay.act(np.zeros(14, dtype=np.float32))
    assert a.shape == (6,)
    assert np.allclose(a[0::2], 0.0)   # angles straight at the enemy goal
    assert np.allclose(a[1::2], 1.0)   # full power


def test_policy_std_scale_affects_sampling_spread():
    net = _fresh_net()
    sd = net.state_dict()
    raw = np.zeros(14, dtype=np.float32)
    # deterministic: std_scale ignored, repeated calls are identical
    det = PolicyOpponent(sd, deterministic=True, hidden=(32, 32), std_scale=0.3)
    assert np.allclose(det.act(raw), det.act(raw))
    # stochastic with a shrunk spread is tighter around the mean than full std
    np.random.seed(0)
    import torch
    torch.manual_seed(0)
    small = PolicyOpponent(sd, deterministic=False, hidden=(32, 32), std_scale=0.3)
    wide = PolicyOpponent(sd, deterministic=False, hidden=(32, 32), std_scale=1.0)
    mean = det.act(raw)
    small_dev = np.std([small.act(raw) - mean for _ in range(200)])
    wide_dev = np.std([wide.act(raw) - mean for _ in range(200)])
    assert small_dev < wide_dev


# ---- LayerNorm snapshots load into scoring/self-play opponents ----

def test_policy_opponent_auto_detects_layernorm_snapshots():
    """
    Docstring for test_policy_opponent_auto_detects_layernorm_snapshots:

    A PolicyOpponent must load a LayerNorm checkpoint even when it was built
    plain (and vice versa): the live-Elo and T5 scorers construct it without
    knowing the subject's architecture, so load() infers LayerNorm from the
    state_dict and rebuilds the net to match. Without this the step-5 run crashes
    the moment the pool or the scorer meets a LayerNorm snapshot.
    """
    ln = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    plain = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=False)
    raw = np.zeros(14, dtype=np.float32)

    po = PolicyOpponent(hidden=(256, 256))       # built plain
    po.load(ln.state_dict())                     # fed a LayerNorm save
    assert po.net.use_layernorm is True
    assert po.act(raw).shape == (6,)
    po.load(plain.state_dict())                  # and back to plain
    assert po.net.use_layernorm is False
    assert po.act(raw).shape == (6,)


def test_self_play_pool_loads_a_layernorm_snapshot():
    """
    Docstring for test_self_play_pool_loads_a_layernorm_snapshot:

    A LayerNorm self-play pool seeded with a LayerNorm snapshot plays it on a
    frozen-copy reset without a shape mismatch.
    """
    ln = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    pool = SelfPlayPool(scripted_prob=0.0, live_prob=0.0, use_layernorm=True, seed=0)
    pool.add(ln.state_dict())
    drew_self = False
    for _ in range(20):
        pool.reset()
        if pool.name == 'self':
            drew_self = True
            assert pool.act(np.zeros(14, dtype=np.float32)).shape == (6,)
    assert drew_self


def test_pool_state_round_trips_use_layernorm():
    """
    Docstring for test_pool_state_round_trips_use_layernorm:

    A saved LayerNorm pool restores as LayerNorm, so a resumed step-5 run loads
    its snapshots into the right architecture.
    """
    ln = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    pool = SelfPlayPool(use_layernorm=True, seed=0)
    pool.add(ln.state_dict())
    restored = SelfPlayPool(use_layernorm=False, seed=0)  # wrong flag on purpose
    restored.load_state(pool.state())
    assert restored.use_layernorm is True
    assert restored.policy.net.use_layernorm is True
    restored.policy.load(restored.snapshots[0])  # must not raise
